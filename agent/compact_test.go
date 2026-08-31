package agent

import (
	"fmt"
	"strings"
	"testing"

	"github.com/momo-mnsjtxy/agentcore/model"
)

func TestCompactKeepsRecentTurnsAndSummarizesOlderMessages(t *testing.T) {
	brain := New(&fakeProvider{}, &fakeTools{}, "test")
	var messages []model.Message
	for turn := 1; turn <= 5; turn++ {
		callID := fmt.Sprintf("call_%d", turn)
		messages = append(messages,
			model.Message{Role: "user", Content: fmt.Sprintf("goal %d", turn)},
			model.Message{Role: "assistant", Content: fmt.Sprintf("answer %d", turn), ToolCalls: []model.ToolCall{{
				ID: callID, Type: "function", Function: model.FunctionCall{Name: "read_file", Arguments: fmt.Sprintf(`{"path":"file%d.go"}`, turn)},
			}}},
			model.Message{Role: "tool", ToolCallID: callID, Content: fmt.Sprintf("result %d", turn)},
		)
	}
	brain.Restore(Snapshot{Messages: messages, Turns: 5})

	result := brain.Compact(2)
	if result.RemovedMessages <= 0 || result.AfterMessages >= result.BeforeMessages {
		t.Fatalf("context was not compacted: %#v", result)
	}
	snapshot := brain.Snapshot()
	if snapshot.Messages[0].Role != "system" || !strings.Contains(snapshot.Messages[0].Content, "goal 1") {
		t.Fatalf("missing factual checkpoint: %#v", snapshot.Messages[0])
	}
	if snapshot.Messages[1].Role != "user" || snapshot.Messages[1].Content != "goal 4" {
		t.Fatalf("recent turns were not preserved at a user boundary: %#v", snapshot.Messages)
	}
	if last := snapshot.Messages[len(snapshot.Messages)-1]; last.Role != "tool" || last.ToolCallID != "call_5" {
		t.Fatalf("recent tool result was detached: %#v", last)
	}
}

func TestCompactDoesNothingWhenConversationIsAlreadyShort(t *testing.T) {
	brain := New(&fakeProvider{}, &fakeTools{}, "test")
	brain.Restore(Snapshot{Messages: []model.Message{
		{Role: "user", Content: "one"},
		{Role: "assistant", Content: "answer"},
	}, Turns: 1})

	result := brain.Compact(4)
	if result.RemovedMessages != 0 || result.BeforeMessages != result.AfterMessages {
		t.Fatalf("short conversation changed: %#v", result)
	}
}

func TestCompactSummaryIsBounded(t *testing.T) {
	brain := New(&fakeProvider{}, &fakeTools{}, "test")
	large := strings.Repeat("context ", compactSummaryLimit)
	brain.Restore(Snapshot{Messages: []model.Message{
		{Role: "user", Content: large},
		{Role: "assistant", Content: large},
		{Role: "user", Content: "recent"},
		{Role: "assistant", Content: "recent answer"},
	}, Turns: 2})

	brain.Compact(1)
	summary := brain.Snapshot().Messages[0].Content
	if len([]rune(summary)) > compactSummaryLimit {
		t.Fatalf("summary has %d runes", len([]rune(summary)))
	}
}
