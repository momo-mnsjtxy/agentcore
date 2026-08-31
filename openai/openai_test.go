package openai

import (
	"strings"
	"testing"

	"github.com/momo-mnsjtxy/agentcore/model"
)

func TestReadBuildsTextAndToolCalls(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"I will inspect."}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_","arguments":"{\"path\":"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"\"README.md\"}"}}]}}]}`,
		`data: [DONE]`,
	}, "\n\n"))

	var deltas strings.Builder
	reply, err := read(stream, func(event model.StreamEvent) {
		if event.Kind == model.OutputDelta {
			deltas.WriteString(event.Text)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if reply.Content != "I will inspect." || deltas.String() != reply.Content {
		t.Fatalf("unexpected content: %#v", reply.Content)
	}
	if len(reply.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(reply.ToolCalls))
	}
	call := reply.ToolCalls[0]
	if call.ID != "call_1" || call.Function.Name != "read_file" || call.Function.Arguments != `{"path":"README.md"}` {
		t.Fatalf("unexpected tool call: %#v", call)
	}
}

func TestChatMessagesAttachImages(t *testing.T) {
	messages := chatMessages([]model.Message{{
		Role:    "tool",
		Content: "see",
		Images:  []model.Image{{MIMEType: "image/png", Data: []byte("png")}},
	}})
	if len(messages) != 2 {
		t.Fatalf("expected tool text plus a follow-up image message, got %#v", messages)
	}
	content, ok := messages[1].(map[string]any)["content"].([]any)
	if !ok || len(content) != 2 {
		t.Fatalf("unexpected image message: %#v", messages[1])
	}
	image, ok := content[1].(map[string]any)["image_url"].(map[string]any)
	if !ok || !strings.HasPrefix(image["url"].(string), "data:image/png;base64,") {
		t.Fatalf("missing data URL: %#v", content[1])
	}
}

func TestReadReturnsStreamError(t *testing.T) {
	_, err := read(strings.NewReader(`data: {"error":{"message":"model unavailable"}}`), func(model.StreamEvent) {})
	if err == nil || !strings.Contains(err.Error(), "model unavailable") {
		t.Fatalf("expected model error, got %v", err)
	}
}
