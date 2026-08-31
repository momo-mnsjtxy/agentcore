/*
OpenAI 模型连接：在 Responses 与 Chat Completions 协议之间路由流式请求。
文字会话按原协议发送；带图的工具结果会编成模型能看的 image part。
连接错误与限流在首个输出事件前有限重试。
调用示例：openai.New(baseURL, apiKey, modelName, protocol, reasoning)。
*/
package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/momo-mnsjtxy/agentcore/model"
)

const maxRequestAttempts = 3

type streamDecoder func(io.Reader, model.Stream) (model.Reply, bool, error)

// Client 连接一个 OpenAI-compatible 服务。
type Client struct {
	baseURL   string
	apiKey    string
	model     string
	protocol  string
	reasoning string
	http      *http.Client
	wait      func(context.Context, time.Duration) error
	jitter    func(time.Duration) time.Duration
}

// New 创建复用连接的流式模型客户端。
func New(baseURL, apiKey, modelName, protocol, reasoning string) *Client {
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          32,
		MaxIdleConnsPerHost:   8,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
	}

	return &Client{
		baseURL:   baseURL,
		apiKey:    apiKey,
		model:     modelName,
		protocol:  protocol,
		reasoning: reasoning,
		http:      &http.Client{Transport: transport},
		wait:      waitContext,
		jitter:    randomDuration,
	}
}

// Complete 流式完成一轮模型推理。
func (client *Client) Complete(context context.Context, messages []model.Message, tools []model.Tool, stream model.Stream) (model.Reply, error) {
	if client.protocol == "responses" {
		return client.completeResponses(context, messages, tools, stream)
	}
	return client.completeChat(context, messages, tools, stream)
}

func (client *Client) completeChat(context context.Context, messages []model.Message, tools []model.Tool, stream model.Stream) (model.Reply, error) {
	payload, err := json.Marshal(map[string]any{
		"model":       client.model,
		"messages":    chatMessages(messages),
		"tools":       tools,
		"tool_choice": "auto",
		"stream_options": map[string]any{
			"include_usage": true,
		},
		"stream":      true,
	})
	if err != nil {
		return model.Reply{}, err
	}

	return client.completeStream(context, "/chat/completions", payload, stream, readChatStream)
}

func read(source io.Reader, stream model.Stream) (model.Reply, error) {
	reply, _, err := readChatStream(source, stream)
	return reply, err
}

func readChatStream(source io.Reader, stream model.Stream) (model.Reply, bool, error) {
	scanner := bufio.NewScanner(source)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)

	var reply model.Reply
	progressed := false
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
			Usage model.Usage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return model.Reply{}, progressed, fmt.Errorf("decode model stream: %w", err)
		}
		if chunk.Error != nil {
			return model.Reply{}, progressed, fmt.Errorf("model stream: %s", chunk.Error.Message)
		}
		if chunk.Usage.TotalTokens > 0 {
			progressed = true
			reply.Usage = chunk.Usage
			stream(model.StreamEvent{Kind: model.UsageChanged, Usage: chunk.Usage})
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta
		if delta.Content != "" {
			progressed = true
			reply.Content += delta.Content
			stream(model.StreamEvent{Kind: model.OutputDelta, Text: delta.Content})
		}
		for _, incoming := range delta.ToolCalls {
			progressed = true
			for len(reply.ToolCalls) <= incoming.Index {
				reply.ToolCalls = append(reply.ToolCalls, model.ToolCall{Type: "function"})
			}
			call := &reply.ToolCalls[incoming.Index]
			if incoming.ID != "" {
				call.ID = incoming.ID
			}
			if incoming.Type != "" {
				call.Type = incoming.Type
			}
			call.Function.Name += incoming.Function.Name
			call.Function.Arguments += incoming.Function.Arguments
		}
	}
	if err := scanner.Err(); err != nil {
		return model.Reply{}, progressed, err
	}
	return reply, progressed, nil
}

func (client *Client) completeStream(runContext context.Context, path string, payload []byte, stream model.Stream, decode streamDecoder) (model.Reply, error) {
	var lastErr error
	for attempt := 0; attempt < maxRequestAttempts; attempt++ {
		request, err := http.NewRequestWithContext(runContext, http.MethodPost, client.baseURL+path, bytes.NewReader(payload))
		if err != nil {
			return model.Reply{}, err
		}
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Accept", "text/event-stream")
		if client.apiKey != "" {
			request.Header.Set("Authorization", "Bearer "+client.apiKey)
		}

		response, err := client.http.Do(request)
		if err != nil {
			if runContext.Err() != nil {
				return model.Reply{}, runContext.Err()
			}
			lastErr = err
			if !retryableConnectionError(err) || attempt+1 == maxRequestAttempts {
				return model.Reply{}, retryError(attempt+1, err)
			}
			if waitErr := client.waitForRetry(runContext, attempt, 0, false); waitErr != nil {
				return model.Reply{}, waitErr
			}
			continue
		}

		if response.StatusCode < 200 || response.StatusCode >= 300 {
			body, _ := io.ReadAll(io.LimitReader(response.Body, 64*1024))
			_ = response.Body.Close()
			lastErr = fmt.Errorf("model returned %s: %s", response.Status, strings.TrimSpace(string(body)))
			if !retryableStatus(response.StatusCode) || attempt+1 == maxRequestAttempts {
				return model.Reply{}, retryError(attempt+1, lastErr)
			}
			delay, present := parseRetryAfter(response.Header.Get("Retry-After"), time.Now())
			if waitErr := client.waitForRetry(runContext, attempt, delay, present); waitErr != nil {
				return model.Reply{}, waitErr
			}
			continue
		}

		reply, progressed, readErr := decode(response.Body, stream)
		_ = response.Body.Close()
		if readErr == nil {
			return reply, nil
		}
		lastErr = readErr
		if progressed || !retryableConnectionError(readErr) || attempt+1 == maxRequestAttempts {
			return model.Reply{}, retryError(attempt+1, readErr)
		}
		if waitErr := client.waitForRetry(runContext, attempt, 0, false); waitErr != nil {
			return model.Reply{}, waitErr
		}
	}
	return model.Reply{}, lastErr
}

func (client *Client) waitForRetry(runContext context.Context, attempt int, delay time.Duration, explicit bool) error {
	if !explicit {
		base := 250 * time.Millisecond * time.Duration(1<<attempt)
		delay = base + client.jitter(base/2)
	}
	return client.wait(runContext, delay)
}

func retryableStatus(status int) bool {
	return status == http.StatusTooManyRequests || status == http.StatusInternalServerError ||
		status == http.StatusBadGateway || status == http.StatusServiceUnavailable || status == http.StatusGatewayTimeout
}

func retryableConnectionError(err error) bool {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	var networkError net.Error
	return errors.As(err, &networkError)
}

func parseRetryAfter(value string, now time.Time) (time.Duration, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	if seconds, err := strconv.Atoi(value); err == nil && seconds >= 0 {
		return time.Duration(seconds) * time.Second, true
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0, false
	}
	delay := when.Sub(now)
	if delay < 0 {
		delay = 0
	}
	return delay, true
}

func retryError(attempts int, err error) error {
	if attempts <= 1 {
		return err
	}
	return fmt.Errorf("model request failed after %d attempts: %w", attempts, err)
}

func waitContext(runContext context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-runContext.Done():
		return runContext.Err()
	}
}

func randomDuration(limit time.Duration) time.Duration {
	if limit <= 0 {
		return 0
	}
	return time.Duration(rand.Int64N(int64(limit) + 1))
}

func chatMessages(messages []model.Message) []any {
	out := make([]any, 0, len(messages)+2)
	for _, message := range messages {
		item := map[string]any{"role": message.Role}
		if message.ToolCallID != "" {
			item["tool_call_id"] = message.ToolCallID
		}
		if len(message.ToolCalls) > 0 {
			item["tool_calls"] = message.ToolCalls
		}
		if len(message.Images) == 0 {
			item["content"] = message.Content
			out = append(out, item)
			continue
		}
		if message.Role == "tool" {
			item["content"] = message.Content
			out = append(out, item)
			out = append(out, map[string]any{"role": "user", "content": imageContent(message.Content, message.Images)})
			continue
		}
		item["content"] = imageContent(message.Content, message.Images)
		out = append(out, item)
	}
	return out
}

func imageContent(text string, images []model.Image) []any {
	parts := make([]any, 0, 1+len(images))
	if text != "" {
		parts = append(parts, map[string]any{"type": "text", "text": text})
	}
	for _, image := range images {
		parts = append(parts, map[string]any{
			"type":      "image_url",
			"image_url": map[string]any{"url": dataURL(image)},
		})
	}
	return parts
}

func dataURL(image model.Image) string {
	mime := image.MIMEType
	if mime == "" {
		mime = "image/png"
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(image.Data)
}
