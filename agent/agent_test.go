package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/momo-mnsjtxy/agentcore/model"
)

func TestRunCompletesApprovedToolLoop(t *testing.T) {
	replies := []model.Reply{
		{ToolCalls: []model.ToolCall{{
			ID: "call_1", Type: "function",
			Function: model.FunctionCall{Name: "run", Arguments: `{"command":"go test ./..."}`},
		}}},
		model.Reply{Content: "All tests pass."},
	}
	provider := &fakeProvider{replies: replies}
	toolbox := &fakeTools{}
	brain := New(provider, toolbox, "test")
	events := make(chan Event, 32)
	approvals := make(chan ApprovalDecision, 1)
	approvals <- ApprovalDecision{Approved: true}

	brain.Run(context.Background(), "verify the project", events, approvals)
	close(events)

	seen := map[EventKind]bool{}
	for event := range events {
		seen[event.Kind] = true
	}
	for _, kind := range []EventKind{Delta, MessageDone, Approval, ToolStarted, ToolDone, Finished} {
		if !seen[kind] {
			t.Fatalf("missing event %s", kind)
		}
	}
	if toolbox.executed != 1 || provider.calls != len(replies) {
		t.Fatalf("expected one tool and two model calls, got %d and %d", toolbox.executed, provider.calls)
	}
}

func TestRunReturnsDenialToModel(t *testing.T) {
	replies := []model.Reply{
		{ToolCalls: []model.ToolCall{{ID: "call_1", Type: "function", Function: model.FunctionCall{Name: "run", Arguments: `{}`}}}},
		model.Reply{Content: "I did not run it."},
	}
	provider := &fakeProvider{replies: replies}
	toolbox := &fakeTools{}
	brain := New(provider, toolbox, "test")
	events := make(chan Event, 32)
	approvals := make(chan ApprovalDecision, 1)
	approvals <- ApprovalDecision{Approved: false}

	brain.Run(context.Background(), "do not run", events, approvals)
	if toolbox.executed != 0 {
		t.Fatal("denied tool was executed")
	}
	found := false
	for _, message := range brain.history {
		if message.Role == "tool" && message.Content == "User denied this action." {
			found = true
		}
	}
	if !found {
		t.Fatal("denial was not returned to the model")
	}
}

func TestRunDefaultsToDenialWithoutApprovalChannel(t *testing.T) {
	replies := []model.Reply{
		{ToolCalls: []model.ToolCall{{ID: "call_1", Type: "function", Function: model.FunctionCall{Name: "run", Arguments: `{}`}}}},
		model.Reply{Content: "I did not run it."},
	}
	toolbox := &fakeTools{}
	brain := New(&fakeProvider{replies: replies}, toolbox, "test")

	brain.Run(context.Background(), "do not wait forever", make(chan Event, 32), nil)

	if toolbox.executed != 0 {
		t.Fatal("tool executed without an approval channel")
	}
}

func TestRunAllowsCallersToIgnoreEvents(t *testing.T) {
	brain := New(&fakeProvider{replies: []model.Reply{{Content: "Done."}}}, &fakeTools{}, "test")
	done := make(chan struct{})
	go func() {
		brain.Run(context.Background(), "finish quietly", nil, nil)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("run blocked on a nil event channel")
	}
}

func TestRunAlwaysReportsCancellation(t *testing.T) {
	brain := New(cancelledProvider{}, &fakeTools{}, "test")
	events := make(chan Event, 4)
	context, cancel := context.WithCancel(context.Background())
	cancel()

	brain.Run(context, "stop", events, make(chan ApprovalDecision))
	event := <-events
	if event.Kind != Finished || event.Text != "Cancelled" {
		t.Fatalf("unexpected cancellation event: %#v", event)
	}
}

func TestRunRemembersExactApprovalForSession(t *testing.T) {
	call := model.ToolCall{ID: "call_1", Type: "function", Function: model.FunctionCall{Name: "run", Arguments: `{"command":"go test ./..."}`}}
	replies := []model.Reply{
		{ToolCalls: []model.ToolCall{call}},
		{ToolCalls: []model.ToolCall{{ID: "call_2", Type: call.Type, Function: call.Function}}},
		model.Reply{Content: "Done."},
	}
	provider := &fakeProvider{replies: replies}
	toolbox := &fakeTools{}
	brain := New(provider, toolbox, "test")
	events := make(chan Event, 32)
	approvals := make(chan ApprovalDecision, 1)
	approvals <- ApprovalDecision{Approved: true, Remember: true}

	brain.Run(context.Background(), "verify twice", events, approvals)
	close(events)
	approvalCount := 0
	for event := range events {
		if event.Kind == Approval {
			approvalCount++
		}
	}
	if approvalCount != 1 || toolbox.executed != 2 {
		t.Fatalf("expected one approval and two executions, got %d and %d", approvalCount, toolbox.executed)
	}
}

func TestRunRequestsAndReturnsClarificationAnswer(t *testing.T) {
	provider := &fakeProvider{replies: []model.Reply{
		{ToolCalls: []model.ToolCall{{
			ID: "clarify_1", Type: "function",
			Function: model.FunctionCall{Name: "request_clarification", Arguments: `{"question":"Which package should I inspect?","options":["agent","tui"]}`},
		}}},
		model.Reply{Content: "Inspecting the agent package."},
	}}
	toolbox := &fakeTools{}
	brain := New(provider, toolbox, "test")
	events := make(chan Event, 32)
	approvals := make(chan ApprovalDecision, 1)
	approvals <- ApprovalDecision{Approved: true, Answer: "agent"}

	brain.Run(context.Background(), "inspect the correct package", events, approvals)
	close(events)

	clarified := false
	answered := false
	for event := range events {
		if event.Kind == Clarification {
			clarified = true
		}
	}
	for _, message := range brain.history {
		if message.Role == "tool" && message.ToolCallID == "clarify_1" && strings.Contains(message.Content, "agent") {
			answered = true
		}
	}
	if !clarified || !answered || toolbox.executed != 0 || provider.calls != 2 {
		t.Fatalf("clarification loop failed: clarified=%v answered=%v executed=%d calls=%d", clarified, answered, toolbox.executed, provider.calls)
	}
}

func TestRunStopsRepeatedFailuresWithClarification(t *testing.T) {
	replies := []model.Reply{
		{ToolCalls: []model.ToolCall{{ID: "call_1", Type: "function", Function: model.FunctionCall{Name: "run", Arguments: `{"command":"bad"}`}}}},
		{ToolCalls: []model.ToolCall{{ID: "call_2", Type: "function", Function: model.FunctionCall{Name: "run", Arguments: `{"command":"bad"}`}}}},
		{ToolCalls: []model.ToolCall{{ID: "call_3", Type: "function", Function: model.FunctionCall{Name: "run", Arguments: `{"command":"bad"}`}}}},
		model.Reply{Content: "Stopping."},
	}
	provider := &fakeProvider{replies: replies}
	toolbox := &repeatedFailTools{}
	brain := New(provider, toolbox, "test")
	events := make(chan Event, 64)
	approvals := make(chan ApprovalDecision, 4)
	approvals <- ApprovalDecision{Approved: true}
	approvals <- ApprovalDecision{Approved: true}
	approvals <- ApprovalDecision{Approved: true}
	approvals <- ApprovalDecision{Approved: true, Answer: "Stop and explain the blocker"}

	brain.Run(context.Background(), "run a failing command", events, approvals)
	close(events)

	clarified := false
	for event := range events {
		if event.Kind == Clarification {
			clarified = true
		}
	}
	if !clarified || toolbox.executed != 3 {
		t.Fatalf("repeated failure did not pause correctly: clarified=%v executed=%d", clarified, toolbox.executed)
	}
}

func TestHandleMetaDispatchesKnownToolsAndPassesUnknown(t *testing.T) {
	brain := New(&fakeProvider{}, &fakeTools{}, "test")
	events := make(chan Event, 8)
	approvals := make(chan ApprovalDecision, 1)
	approvals <- ApprovalDecision{Approved: true}

	handled, stop := brain.handleMeta(context.Background(), events, approvals,
		model.ToolCall{ID: "clarify_1", Type: "function", Function: model.FunctionCall{Name: "request_clarification", Arguments: `{"question":"Which?"}`}})
	if !handled || stop {
		t.Fatalf("request_clarification should be handled without stopping: handled=%v stop=%v", handled, stop)
	}

	handled, stop = brain.handleMeta(context.Background(), events, make(chan ApprovalDecision),
		model.ToolCall{ID: "call_1", Type: "function", Function: model.FunctionCall{Name: "run", Arguments: `{}`}})
	if handled {
		t.Fatal("workspace tool should not be handled as a meta tool")
	}
}

func TestStatsTrackToolsApprovalsAndFailures(t *testing.T) {
	replies := []model.Reply{
		{ToolCalls: []model.ToolCall{{ID: "call_1", Type: "function", Function: model.FunctionCall{Name: "run", Arguments: `{"command":"bad"}`}}}},
		{ToolCalls: []model.ToolCall{{ID: "call_2", Type: "function", Function: model.FunctionCall{Name: "run", Arguments: `{"command":"ok"}`}}}},
		model.Reply{Content: "Done."},
	}
	provider := &fakeProvider{replies: replies}
	toolbox := &oneFailureTools{}
	brain := New(provider, toolbox, "test")
	events := make(chan Event, 32)
	approvals := make(chan ApprovalDecision, 2)
	approvals <- ApprovalDecision{Approved: true}
	approvals <- ApprovalDecision{Approved: true}

	brain.Run(context.Background(), "verify", events, approvals)
	close(events)

	stats := brain.Stats()
	if stats.Tools != 2 {
		t.Fatalf("tools=%d want 2", stats.Tools)
	}
	if stats.Approvals != 2 {
		t.Fatalf("approvals=%d want 2", stats.Approvals)
	}
	if stats.Failures != 1 {
		t.Fatalf("failures=%d want 1", stats.Failures)
	}
}

func TestRunStopsWhenStalled(t *testing.T) {
	empty := func() model.Reply {
		return model.Reply{ToolCalls: []model.ToolCall{{ID: "noop_1", Type: "function", Function: model.FunctionCall{Name: "blocked_tool", Arguments: `{}`}}}}
	}
	provider := &fakeProvider{replies: []model.Reply{empty(), empty(), empty(), empty()}}
	toolbox := &blockedTools{}
	brain := New(provider, toolbox, "test")
	events := make(chan Event, 32)

	// 无审批通道默认拒绝，每次调用都被拒而不执行，也没有文本，三轮无进展即停。
	brain.Run(context.Background(), "spin forever", events, nil)
	close(events)

	failed := false
	for event := range events {
		if event.Kind == Failed && strings.Contains(event.Text, "no progress") {
			failed = true
		}
	}
	if !failed {
		t.Fatal("stalled run was not stopped")
	}
	if provider.calls != maxStalledTurns {
		t.Fatalf("expected %d model calls before stopping, got %d", maxStalledTurns, provider.calls)
	}
}

func TestRunStopsAtReasoningTurnLimit(t *testing.T) {
	provider := &loopingProvider{}
	brain := New(provider, &fakeTools{}, "test")
	events := make(chan Event, maxTurns*4+1)

	brain.Run(context.Background(), "keep looping", events, nil)
	close(events)

	var failure Event
	for event := range events {
		if event.Kind == Failed {
			failure = event
		}
	}
	if provider.calls != maxTurns {
		t.Fatalf("model called %d times, want %d", provider.calls, maxTurns)
	}
	if !strings.Contains(failure.Text, fmt.Sprintf("%d reasoning turns", maxTurns)) {
		t.Fatalf("unexpected turn-limit failure: %#v", failure)
	}
}

func TestSnapshotIsDetachedAndRestoreClearsApprovals(t *testing.T) {
	brain := New(&fakeProvider{}, &fakeTools{}, "test")
	brain.Restore(Snapshot{
		Messages: []model.Message{{
			Role:   "assistant",
			Images: []model.Image{{MIMEType: "image/png", Data: []byte("image")}},
			ToolCalls: []model.ToolCall{{
				ID: "call_1", Function: model.FunctionCall{Name: "run", Arguments: `{}`},
			}},
		}},
		Turns: 3,
	})
	brain.rememberApproval("run\x00go test ./...")

	snapshot := brain.Snapshot()
	snapshot.Messages[0].ToolCalls[0].Function.Name = "changed"
	snapshot.Messages[0].Images[0].Data[0] = 'X'
	if current := brain.Snapshot(); current.Messages[0].ToolCalls[0].Function.Name != "run" {
		t.Fatal("snapshot shares tool call storage with the agent")
	}
	if current := brain.Snapshot(); string(current.Messages[0].Images[0].Data) != "image" {
		t.Fatal("snapshot shares image bytes with the agent")
	}

	brain.Restore(snapshot)
	if brain.isApproved("run\x00go test ./...") {
		t.Fatal("restore retained an approval from the previous process")
	}
	if stats := brain.Stats(); stats.Turns != 3 || stats.Messages != 1 {
		t.Fatalf("unexpected restored stats: %#v", stats)
	}
}

func TestRunKeepsToolImagesForTheModel(t *testing.T) {
	picture := model.Image{MIMEType: "image/png", Data: []byte("png")}
	provider := &fakeProvider{replies: []model.Reply{
		{ToolCalls: []model.ToolCall{{ID: "call_1", Type: "function", Function: model.FunctionCall{Name: "screenshot", Arguments: `{}`}}}},
		model.Reply{Content: "I can see the screen."},
	}}
	toolbox := &fakeTools{images: []model.Image{picture}}
	brain := New(provider, toolbox, "test")
	events := make(chan Event, 16)
	approvals := make(chan ApprovalDecision, 1)
	approvals <- ApprovalDecision{Approved: true}

	brain.Run(context.Background(), "look", events, approvals)
	found := false
	for _, message := range brain.history {
		if message.Role == "tool" && len(message.Images) == 1 && string(message.Images[0].Data) == "png" {
			found = true
		}
	}
	if !found {
		t.Fatal("tool image was not returned to the model")
	}
}

func TestAppendMessageDropsOlderScreenshots(t *testing.T) {
	brain := New(&fakeProvider{}, &fakeTools{}, "test")
	for turn := 1; turn <= 5; turn++ {
		brain.appendMessage(model.Message{
			Role:    "tool",
			Content: fmt.Sprintf("shot %d", turn),
			Images:  []model.Image{{MIMEType: "image/png", Data: []byte{byte(turn)}}},
		})
	}
	kept := 0
	for _, message := range brain.history {
		if len(message.Images) > 0 {
			kept++
			if message.Images[0].Data[0] < 3 {
				t.Fatalf("old screenshot was kept: %#v", message)
			}
		}
	}
	if kept != keepRecentImages {
		t.Fatalf("kept %d screenshots, want %d", kept, keepRecentImages)
	}
}

func TestWorkspaceCapabilitiesBridgeOptionalToolboxOperations(t *testing.T) {
	tools := &workspaceTools{}
	brain := New(&fakeProvider{}, tools, "test")

	diff, err := brain.WorkspaceDiff(context.Background())
	if err != nil || diff != "workspace diff" {
		t.Fatalf("diff=%q err=%v", diff, err)
	}
	undone, err := brain.UndoLastChange()
	if err != nil || undone != "change undone" {
		t.Fatalf("undo=%q err=%v", undone, err)
	}
}

type fakeProvider struct {
	replies     []model.Reply
	definitions []model.Tool
	calls       int
}

func (provider *fakeProvider) Complete(_ context.Context, _ []model.Message, definitions []model.Tool, stream model.Stream) (model.Reply, error) {
	provider.definitions = append([]model.Tool(nil), definitions...)
	reply := provider.replies[provider.calls]
	provider.calls++
	if reply.Content != "" {
		stream(model.StreamEvent{Kind: model.OutputDelta, Text: reply.Content})
	}
	return reply, nil
}

type fakeTools struct {
	executed int
	images   []model.Image
}

func (tools *fakeTools) Definitions() []model.Tool { return nil }
func (tools *fakeTools) Capability(call model.ToolCall) Capability {
	switch call.Function.Name {
	case "write_file":
		return CapabilityWrite
	case "run":
		return CapabilityCommand
	default:
		return CapabilityRead
	}
}
func (tools *fakeTools) NeedsApproval(model.ToolCall) bool { return true }
func (tools *fakeTools) Preview(model.ToolCall) string     { return "preview" }
func (tools *fakeTools) Execute(context.Context, model.ToolCall) (model.Observation, error) {
	tools.executed++
	return model.Observation{Text: "ok", Images: tools.images}, nil
}

type blockedTools struct{}

func (*blockedTools) Definitions() []model.Tool { return nil }
func (*blockedTools) Capability(model.ToolCall) Capability {
	return CapabilityWrite
}
func (*blockedTools) NeedsApproval(model.ToolCall) bool { return true }
func (*blockedTools) Preview(model.ToolCall) string     { return "blocked" }
func (*blockedTools) Execute(context.Context, model.ToolCall) (model.Observation, error) {
	return model.Observation{Text: "not allowed"}, nil
}

type oneFailureTools struct {
	executed int
}

func (*oneFailureTools) Definitions() []model.Tool            { return nil }
func (*oneFailureTools) Capability(model.ToolCall) Capability { return CapabilityCommand }
func (*oneFailureTools) NeedsApproval(model.ToolCall) bool    { return true }
func (*oneFailureTools) Preview(model.ToolCall) string        { return "run" }
func (tools *oneFailureTools) Execute(_ context.Context, call model.ToolCall) (model.Observation, error) {
	tools.executed++
	if tools.executed == 1 {
		return model.Observation{Text: "boom"}, errors.New("boom")
	}
	return model.Observation{Text: "ok"}, nil
}

type repeatedFailTools struct {
	executed int
}

func (*repeatedFailTools) Definitions() []model.Tool            { return nil }
func (*repeatedFailTools) Capability(model.ToolCall) Capability { return CapabilityCommand }
func (*repeatedFailTools) NeedsApproval(model.ToolCall) bool    { return true }
func (*repeatedFailTools) Preview(model.ToolCall) string        { return "bad" }
func (tools *repeatedFailTools) Execute(context.Context, model.ToolCall) (model.Observation, error) {
	tools.executed++
	return model.Observation{Text: "boom"}, errors.New("boom")
}

type cancelledProvider struct{}

func (cancelledProvider) Complete(context.Context, []model.Message, []model.Tool, model.Stream) (model.Reply, error) {
	return model.Reply{}, context.Canceled
}

type loopingProvider struct {
	calls int
}

func (provider *loopingProvider) Complete(context.Context, []model.Message, []model.Tool, model.Stream) (model.Reply, error) {
	provider.calls++
	return model.Reply{
		Content: "still considering",
		ToolCalls: []model.ToolCall{{
			ID:   fmt.Sprintf("call_%d", provider.calls),
			Type: "function",
			Function: model.FunctionCall{
				Name:      "looping_tool",
				Arguments: `{}`,
			},
		}},
	}, nil
}

type workspaceTools struct {
	fakeTools
}

func (*workspaceTools) Diff(context.Context) (string, error) { return "workspace diff", nil }
func (*workspaceTools) Undo() (string, error)                { return "change undone", nil }

var _ = json.Marshal
