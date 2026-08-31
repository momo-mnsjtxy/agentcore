package openai

import (
	"strings"
	"testing"

	"github.com/momo-mnsjtxy/agentcore/model"
)

func TestResponsesRequestAndStream(t *testing.T) {
	client := New("https://api.example.test/v1", "secret", "gpt-test", "responses", "high")
	messages := []model.Message{
		{Role: "system", Content: "system rules"},
		{Role: "user", Content: "inspect"},
	}
	tools := []model.Tool{{Type: "function", Function: model.FunctionTool{
		Name: "read_file", Description: "Read file", Parameters: map[string]any{"type": "object"},
	}}}
	payload := client.responsesPayload(messages, tools)
	if payload["instructions"] != "system rules" || payload["stream"] != true || payload["store"] != false {
		t.Fatalf("unexpected request controls: %#v", payload)
	}
	if reasoning, ok := payload["reasoning"].(map[string]any); !ok || reasoning["effort"] != "high" {
		t.Fatalf("unexpected reasoning: %#v", payload["reasoning"])
	}
	requestTools, ok := payload["tools"].([]any)
	if !ok || len(requestTools) != 1 || requestTools[0].(map[string]any)["name"] != "read_file" {
		t.Fatalf("unexpected tools: %#v", payload["tools"])
	}

	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.reasoning_summary_text.delta","delta":"Checking files"}`,
		`data: {"type":"response.output_text.delta","delta":"I will inspect."}`,
		`data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read_file","arguments":""}}`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"path\":\"README.md\"}"}`,
		`data: {"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"README.md\"}"}}`,
		`data: {"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":120,"output_tokens":30,"total_tokens":150}}}`,
		`data: [DONE]`,
	}, "\n\n"))
	var output, reasoning strings.Builder
	reply, err := readResponses(stream, func(event model.StreamEvent) {
		switch event.Kind {
		case model.OutputDelta:
			output.WriteString(event.Text)
		case model.ReasoningDelta:
			reasoning.WriteString(event.Text)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.String() != "I will inspect." || reasoning.String() != "Checking files" {
		t.Fatalf("unexpected stream output=%q reasoning=%q", output.String(), reasoning.String())
	}
	if reply.ResponseID != "resp_1" || reply.Usage.TotalTokens != 150 {
		t.Fatalf("unexpected completion: %#v", reply)
	}
	if len(reply.ToolCalls) != 1 || reply.ToolCalls[0].Function.Arguments != `{"path":"README.md"}` {
		t.Fatalf("unexpected tool calls: %#v", reply.ToolCalls)
	}
}

func TestResponsesPayloadAttachesToolImages(t *testing.T) {
	client := New("https://api.example.test/v1", "", "gpt-test", "responses", "none")
	payload := client.responsesPayload([]model.Message{{
		Role:       "tool",
		ToolCallID: "call_1",
		Content:    "see",
		Images:     []model.Image{{MIMEType: "image/jpeg", Data: []byte("jpg")}},
	}}, nil)
	input := payload["input"].([]any)
	output := input[0].(map[string]any)["output"].([]any)
	if output[0].(map[string]any)["type"] != "input_text" {
		t.Fatalf("unexpected tool output: %#v", output)
	}
	if output[1].(map[string]any)["type"] != "input_image" {
		t.Fatalf("missing tool image: %#v", output)
	}
}

func TestResponsesIncompleteIsAnError(t *testing.T) {
	stream := strings.NewReader("data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n")
	_, err := readResponses(stream, func(model.StreamEvent) {})
	if err == nil || !strings.Contains(err.Error(), "max_output_tokens") {
		t.Fatalf("expected incomplete response error, got %v", err)
	}
}

func TestResponsesUsesCompletedOutputWhenDoneEventIsMissing(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"run","arguments":""}}`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"command\":\"go test"}`,
		`data: {"type":"response.completed","response":{"id":"resp_1","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"run","arguments":"{\"command\":\"go test ./...\"}"}]}}`,
	}, "\n\n"))

	reply, err := readResponses(stream, func(model.StreamEvent) {})
	if err != nil {
		t.Fatal(err)
	}
	if len(reply.ToolCalls) != 1 || reply.ToolCalls[0].Function.Arguments != `{"command":"go test ./..."}` {
		t.Fatalf("unexpected tool calls: %#v", reply.ToolCalls)
	}
}
