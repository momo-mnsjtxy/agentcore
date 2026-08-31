/*
Responses 协议：把会话转换为 input items，并解析文本、推理摘要、工具和用量事件。
该实现保持 store=false，每轮显式发送当前会话，便于连接兼容服务和本地代理。
带图的工具结果会编成 input_image 供模型查看。
*/
package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/momo-mnsjtxy/agentcore/model"
)

func (client *Client) completeResponses(runContext context.Context, messages []model.Message, tools []model.Tool, stream model.Stream) (model.Reply, error) {
	payload := client.responsesPayload(messages, tools)

	encoded, err := json.Marshal(payload)
	if err != nil {
		return model.Reply{}, err
	}
	return client.completeStream(runContext, "/responses", encoded, stream, readResponsesStream)
}

func (client *Client) responsesPayload(messages []model.Message, tools []model.Tool) map[string]any {
	payload := map[string]any{
		"model":               client.model,
		"instructions":        responseInstructions(messages),
		"input":               responseInput(messages),
		"tools":               responseTools(tools),
		"tool_choice":         "auto",
		"parallel_tool_calls": false,
		"store":               false,
		"stream":              true,
	}
	if client.reasoning != "" && client.reasoning != "none" {
		payload["reasoning"] = map[string]any{"effort": client.reasoning, "summary": "auto"}
	}
	return payload
}

func responseInstructions(messages []model.Message) string {
	var instructions []string
	for _, message := range messages {
		if message.Role == "system" && message.Content != "" {
			instructions = append(instructions, message.Content)
		}
	}
	return strings.Join(instructions, "\n\n")
}

func responseInput(messages []model.Message) []any {
	input := make([]any, 0, len(messages)*2)
	for _, message := range messages {
		switch message.Role {
		case "user":
			input = append(input, map[string]any{
				"type": "message", "role": "user",
				"content": responseContent("input_text", message),
			})
		case "assistant":
			if message.Content != "" || len(message.Images) > 0 {
				input = append(input, map[string]any{
					"type": "message", "role": "assistant",
					"content": responseContent("output_text", message),
				})
			}
			for _, call := range message.ToolCalls {
				input = append(input, map[string]any{
					"type": "function_call", "call_id": call.ID,
					"name": call.Function.Name, "arguments": call.Function.Arguments,
				})
			}
		case "tool":
			input = append(input, map[string]any{
				"type": "function_call_output", "call_id": message.ToolCallID, "output": responseOutput(message),
			})
		}
	}
	return input
}

func responseContent(textType string, message model.Message) []any {
	parts := make([]any, 0, 1+len(message.Images))
	if message.Content != "" {
		parts = append(parts, map[string]any{"type": textType, "text": message.Content})
	}
	for _, image := range message.Images {
		parts = append(parts, map[string]any{"type": "input_image", "image_url": dataURL(image)})
	}
	if len(parts) == 0 {
		return []any{map[string]any{"type": textType, "text": ""}}
	}
	return parts
}

func responseOutput(message model.Message) any {
	if len(message.Images) == 0 {
		return message.Content
	}
	parts := []any{map[string]any{"type": "input_text", "text": message.Content}}
	for _, image := range message.Images {
		parts = append(parts, map[string]any{"type": "input_image", "image_url": dataURL(image)})
	}
	return parts
}

func responseTools(tools []model.Tool) []any {
	result := make([]any, 0, len(tools))
	for _, tool := range tools {
		result = append(result, map[string]any{
			"type":        "function",
			"name":        tool.Function.Name,
			"description": tool.Function.Description,
			"parameters":  tool.Function.Parameters,
			"strict":      false,
		})
	}
	return result
}

func readResponses(source io.Reader, stream model.Stream) (model.Reply, error) {
	reply, _, err := readResponsesStream(source, stream)
	return reply, err
}

func readResponsesStream(source io.Reader, stream model.Stream) (model.Reply, bool, error) {
	scanner := bufio.NewScanner(source)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)

	var reply model.Reply
	progressed := false
	calls := map[string]model.ToolCall{}
	var order []string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}

		var event struct {
			Type     string       `json:"type"`
			Delta    string       `json:"delta"`
			ItemID   string       `json:"item_id"`
			Item     responseItem `json:"item"`
			Response struct {
				ID                string         `json:"id"`
				Usage             model.Usage    `json:"usage"`
				Output            []responseItem `json:"output"`
				Error             *apiError      `json:"error"`
				IncompleteDetails *struct {
					Reason string `json:"reason"`
				} `json:"incomplete_details"`
			} `json:"response"`
			Error *apiError `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return model.Reply{}, progressed, fmt.Errorf("decode responses stream: %w", err)
		}

		switch event.Type {
		case "response.output_text.delta":
			progressed = true
			reply.Content += event.Delta
			stream(model.StreamEvent{Kind: model.OutputDelta, Text: event.Delta})
		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			progressed = true
			stream(model.StreamEvent{Kind: model.ReasoningDelta, Text: event.Delta})
		case "response.output_item.added":
			if event.Item.Type == "function_call" {
				progressed = true
				mergeResponseCall(calls, &order, event.Item)
			}
		case "response.function_call_arguments.delta":
			progressed = true
			if _, exists := calls[event.ItemID]; !exists && event.ItemID != "" {
				order = append(order, event.ItemID)
			}
			call := calls[event.ItemID]
			call.Function.Arguments += event.Delta
			calls[event.ItemID] = call
		case "response.output_item.done":
			if event.Item.Type == "function_call" {
				progressed = true
				mergeResponseCall(calls, &order, event.Item)
			}
		case "response.completed":
			progressed = true
			for _, item := range event.Response.Output {
				if item.Type == "function_call" {
					mergeResponseCall(calls, &order, item)
				}
			}
			reply.ResponseID = event.Response.ID
			reply.Usage = event.Response.Usage
			if reply.Usage.TotalTokens > 0 {
				stream(model.StreamEvent{Kind: model.UsageChanged, Usage: reply.Usage})
			}
		case "response.failed":
			return model.Reply{}, progressed, event.Response.Error.asError("response failed")
		case "response.incomplete":
			reason := "unknown"
			if event.Response.IncompleteDetails != nil && event.Response.IncompleteDetails.Reason != "" {
				reason = event.Response.IncompleteDetails.Reason
			}
			return model.Reply{}, progressed, fmt.Errorf("incomplete response: %s", reason)
		case "error":
			return model.Reply{}, progressed, event.Error.asError("responses stream failed")
		}
	}
	if err := scanner.Err(); err != nil {
		return model.Reply{}, progressed, err
	}
	for _, key := range order {
		call := calls[key]
		if call.ID != "" && call.Function.Name != "" {
			reply.ToolCalls = append(reply.ToolCalls, call)
		}
	}
	return reply, progressed, nil
}

type responseItem struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type apiError struct {
	Message string `json:"message"`
	Code    string `json:"code"`
}

func (failure *apiError) asError(fallback string) error {
	if failure == nil {
		return fmt.Errorf("%s", fallback)
	}
	if failure.Code != "" {
		return fmt.Errorf("%s: %s (%s)", fallback, failure.Message, failure.Code)
	}
	return fmt.Errorf("%s: %s", fallback, failure.Message)
}

func responseCallKey(itemID, callID string) string {
	if itemID != "" {
		return itemID
	}
	return callID
}

func mergeResponseCall(calls map[string]model.ToolCall, order *[]string, item responseItem) {
	key := responseCallKey(item.ID, item.CallID)
	if key == "" {
		return
	}
	call, exists := calls[key]
	if !exists {
		*order = append(*order, key)
		call.Type = "function"
	}
	if item.CallID != "" {
		call.ID = item.CallID
	}
	if item.Name != "" {
		call.Function.Name = item.Name
	}
	if item.Arguments != "" {
		call.Function.Arguments = item.Arguments
	}
	calls[key] = call
}
