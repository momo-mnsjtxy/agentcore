/*
Agent 认知核心：接收输入，让模型决定下一步，执行工具并把观察结果送回模型。
循环只做四件事：推理、工具、观察、继续。不含交付工作流、任务台账或产出机制——
那些属于上层的 agentengine，由应用以 Toolbox 和系统提示的身份接入。
调用示例：agent.New(provider, tools, system).Run(context, input, events, approvals)。
*/
package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/momo-mnsjtxy/agentcore/model"
)

const (
	keepRecentImages = 3  // 旧截图对当前画面几乎没价值，却会占满上下文
	maxTurns         = 32 // 即使每轮都有形式进展，也不能让单次运行无限消耗模型调用
	maxStalledTurns  = 3  // 连续三轮既没执行工具、也没产出文本就判定停滞
)

// EventKind 描述认知循环正在发生的业务动作。
type EventKind string

const (
	Delta        EventKind = "delta"
	Reasoning    EventKind = "reasoning"
	MessageDone  EventKind = "message_done"
	Approval     EventKind = "approval"
	ToolStarted  EventKind = "tool_started"
	ToolDone     EventKind = "tool_done"
	UsageChanged EventKind = "usage_changed"
	Clarification EventKind = "clarification"
	Finished     EventKind = "finished"
	Failed       EventKind = "failed"
)

// Event 是认知循环向界面发送的一次变化。
type Event struct {
	Kind  EventKind
	Text  string
	Call  model.ToolCall
	Usage model.Usage
	Failed bool
}

// ApprovalDecision 是用户对一次高风险动作的决定。
type ApprovalDecision struct {
	Approved bool
	Remember bool
	Answer   string
}

// Stats 是当前会话的轻量运行数据，供应用展示。
type Stats struct {
	Turns     int
	Messages  int
	Tools     int
	Approvals int
	Failures  int
}

// Snapshot 是可持久化的会话状态，不包含系统提示和审批许可。
type Snapshot struct {
	Messages []model.Message `json:"messages"`
	Turns    int             `json:"turns"`
}

// Capability 描述工具对环境的改动程度，由 Toolbox 声明。
type Capability string

const (
	CapabilityRead    Capability = "read"    // 只读观察，始终放行
	CapabilityWrite   Capability = "write"   // 修改文件等副作用
	CapabilityCommand Capability = "command" // 运行命令
)

// Toolbox 是 Agent 可以观察和改变环境的能力。这五个方法是必须的。
type Toolbox interface {
	Definitions() []model.Tool
	Capability(model.ToolCall) Capability
	NeedsApproval(model.ToolCall) bool
	Preview(model.ToolCall) string
	Execute(context.Context, model.ToolCall) (model.Observation, error)
}

// WorkspaceDiffer 让 Toolbox 交出完整的环境 diff。
type WorkspaceDiffer interface {
	Diff(context.Context) (string, error)
}

// WorkspaceUndoer 让 Toolbox 回退最近一次结构化改动。
type WorkspaceUndoer interface {
	Undo() (string, error)
}

// Agent 持有当前会话和完整认知循环。
type Agent struct {
	mu            sync.RWMutex
	provider      model.Provider
	tools         Toolbox
	system        model.Message
	history       []model.Message
	approved      map[string]bool
	turns         int
	toolRuns      int
	approvals     int
	toolFailures  int
	failureKey    string
	failureStreak int
}

// DefaultSystem 是核心的通用行为提示。应用应在其上叠加产品身份与项目指令。
func DefaultSystem() string {
	return `Complete the user's goal end to end. Inspect relevant files before editing. Keep the main business flow obvious, make the smallest coherent change, and verify important behavior after edits. Use tools whenever evidence is available locally. Continue through tool results until the task is genuinely complete. Explain blockers plainly instead of pretending success.

Read tools are automatic. Commands and file changes require user approval. Never claim an action happened unless its tool result confirms it.`
}

// New 创建一个带系统提示的新会话。提示由应用提供，核心不假设工作区或产品身份。
func New(provider model.Provider, tools Toolbox, system string) *Agent {
	if strings.TrimSpace(system) == "" {
		system = DefaultSystem()
	}
	message := model.Message{Role: "system", Content: system}

	return &Agent{
		provider: provider,
		tools:    tools,
		system:   message,
		history:  []model.Message{message},
		approved: map[string]bool{},
	}
}

// Reset 清空会话并保留同一套项目指令。
func (agent *Agent) Reset() {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.history = []model.Message{agent.system}
	agent.approved = map[string]bool{}
	agent.turns = 0
	agent.toolRuns = 0
	agent.approvals = 0
	agent.toolFailures = 0
	agent.failureKey = ""
	agent.failureStreak = 0
}

// Stats reports current conversation size and activity without exposing message contents.
func (agent *Agent) Stats() Stats {
	agent.mu.RLock()
	defer agent.mu.RUnlock()
	return Stats{
		Turns:     agent.turns,
		Messages:  len(agent.history) - 1,
		Tools:     agent.toolRuns,
		Approvals: agent.approvals,
		Failures:  agent.toolFailures,
	}
}

// Snapshot returns a detached conversation copy safe for persistence.
func (agent *Agent) Snapshot() Snapshot {
	agent.mu.RLock()
	defer agent.mu.RUnlock()
	return Snapshot{Messages: cloneMessages(agent.history[1:]), Turns: agent.turns}
}

// Restore replaces conversation state while intentionally clearing approvals.
func (agent *Agent) Restore(snapshot Snapshot) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.history = append([]model.Message{agent.system}, cloneMessages(snapshot.Messages)...)
	agent.turns = max(0, snapshot.Turns)
	agent.approved = map[string]bool{}
	agent.toolRuns = 0
	agent.approvals = 0
	agent.toolFailures = 0
	agent.failureKey = ""
	agent.failureStreak = 0
}

// WorkspaceDiff returns the complete inspectable workspace diff when supported by the toolbox.
func (agent *Agent) WorkspaceDiff(runContext context.Context) (string, error) {
	differ, ok := agent.tools.(WorkspaceDiffer)
	if !ok {
		return "", errors.New("workspace diff is unavailable")
	}
	return differ.Diff(runContext)
}

// UndoLastChange reverts the latest structured file mutation when supported by the toolbox.
func (agent *Agent) UndoLastChange() (string, error) {
	undoer, ok := agent.tools.(WorkspaceUndoer)
	if !ok {
		return "", errors.New("workspace undo is unavailable")
	}
	return undoer.Undo()
}

// Run pursues one user input until the model finishes or the context is cancelled.
func (agent *Agent) Run(runContext context.Context, input string, events chan<- Event, approvals <-chan ApprovalDecision) {
	input = strings.TrimSpace(input)
	if input == "" {
		send(runContext, events, Event{Kind: Finished})
		return
	}
	agent.appendMessage(model.Message{Role: "user", Content: input})

	stalled := 0
	for turn := 0; turn < maxTurns; turn++ {
		history := agent.startTurn()
		reply, err := agent.provider.Complete(runContext, history, agent.toolDefinitions(), func(event model.StreamEvent) {
			switch event.Kind {
			case model.OutputDelta:
				send(runContext, events, Event{Kind: Delta, Text: event.Text})
			case model.ReasoningDelta:
				send(runContext, events, Event{Kind: Reasoning, Text: event.Text})
			case model.UsageChanged:
				send(runContext, events, Event{Kind: UsageChanged, Usage: event.Usage})
			}
		})
		if err != nil {
			if errors.Is(err, context.Canceled) {
				send(runContext, events, Event{Kind: Finished, Text: "Cancelled"})
				return
			}
			send(runContext, events, Event{Kind: Failed, Text: err.Error(), Failed: true})
			return
		}

		agent.appendMessage(model.Message{Role: "assistant", Content: reply.Content, ToolCalls: reply.ToolCalls})
		if reply.Usage.TotalTokens > 0 {
			send(runContext, events, Event{Kind: UsageChanged, Usage: reply.Usage})
		}
		send(runContext, events, Event{Kind: MessageDone})
		if len(reply.ToolCalls) == 0 {
			send(runContext, events, Event{Kind: Finished})
			return
		}

		progressed := reply.Content != ""
		for _, call := range reply.ToolCalls {
			if handled, stop := agent.handleMeta(runContext, events, approvals, call); handled {
				progressed = true
				if stop {
					return
				}
				continue
			}

			preview := agent.tools.Preview(call)
			key := call.Function.Name + "\x00" + preview
			approved := true
			if agent.tools.NeedsApproval(call) && !agent.isApproved(key) {
				agent.countApproval()
				send(runContext, events, Event{Kind: Approval, Text: preview, Call: call})
				decision, received := receiveDecision(runContext, approvals)
				if !received {
					send(runContext, events, Event{Kind: Finished, Text: "Cancelled"})
					return
				}
				approved = decision.Approved
				if decision.Approved && decision.Remember {
					agent.rememberApproval(key)
				}
			}

			if !approved {
				agent.refuseCall(runContext, events, call, "User denied this action.")
				continue
			}

			send(runContext, events, Event{Kind: ToolStarted, Text: preview, Call: call})
			observation, runErr := agent.tools.Execute(runContext, call)
			agent.countToolRun(runErr != nil)
			progressed = true
			output := observation.Text
			if runErr != nil {
				output = strings.TrimSpace(output + "\nError: " + runErr.Error())
			}
			agent.appendMessage(model.Message{Role: "tool", ToolCallID: call.ID, Content: output, Images: observation.Images})
			if agent.recordToolOutcome(call, output, runErr != nil) {
				if agent.requestRepeatedFailureClarification(runContext, events, approvals, call, output) {
					return
				}
			}
			send(runContext, events, Event{Kind: ToolDone, Text: output, Call: call, Failed: runErr != nil})
		}

		if progressed {
			stalled = 0
		} else {
			stalled++
			if stalled >= maxStalledTurns {
				send(runContext, events, Event{Kind: Failed, Text: "Agent made no progress for several turns.", Failed: true})
				return
			}
		}
	}

	send(runContext, events, Event{Kind: Failed, Text: fmt.Sprintf("Agent stopped after %d reasoning turns.", maxTurns), Failed: true})
}

// toolDefinitions returns the application's tools. The core owns no reserved names.
func (agent *Agent) toolDefinitions() []model.Tool {
	return agent.tools.Definitions()
}

// refuseCall reports a call the core will not run — denied by the user.
func (agent *Agent) refuseCall(runContext context.Context, events chan<- Event, call model.ToolCall, reason string) {
	agent.appendMessage(model.Message{Role: "tool", ToolCallID: call.ID, Content: reason})
	agent.recordToolOutcome(call, reason, false)
	send(runContext, events, Event{Kind: ToolDone, Text: reason, Call: call, Failed: false})
}

func (agent *Agent) recordToolOutcome(call model.ToolCall, output string, failed bool) bool {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	if failed {
		key := call.Function.Name + "\x00" + firstTaskLine(output)
		if key == agent.failureKey {
			agent.failureStreak++
		} else {
			agent.failureKey = key
			agent.failureStreak = 1
		}
	} else {
		agent.failureKey = ""
		agent.failureStreak = 0
	}
	return agent.failureStreak >= 3
}

func (agent *Agent) appendMessage(message model.Message) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.history = append(agent.history, message)
	if len(message.Images) > 0 {
		pruneImages(agent.history, keepRecentImages)
	}
}

func (agent *Agent) startTurn() []model.Message {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.turns++
	return cloneMessages(agent.history)
}

func (agent *Agent) isApproved(key string) bool {
	agent.mu.RLock()
	defer agent.mu.RUnlock()
	return agent.approved[key]
}

func (agent *Agent) rememberApproval(key string) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.approved[key] = true
}

func (agent *Agent) countApproval() {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.approvals++
}

func (agent *Agent) countToolRun(failed bool) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.toolRuns++
	if failed {
		agent.toolFailures++
	}
}

func pruneImages(messages []model.Message, keep int) {
	if keep < 1 {
		keep = 1
	}
	seen := 0
	for index := len(messages) - 1; index >= 0; index-- {
		if len(messages[index].Images) == 0 {
			continue
		}
		seen++
		if seen > keep {
			messages[index].Images = nil
		}
	}
}

func cloneMessages(messages []model.Message) []model.Message {
	cloned := make([]model.Message, len(messages))
	for index, message := range messages {
		cloned[index] = message
		cloned[index].ToolCalls = append([]model.ToolCall(nil), message.ToolCalls...)
		if len(message.Images) > 0 {
			cloned[index].Images = make([]model.Image, len(message.Images))
			for imageIndex, image := range message.Images {
				cloned[index].Images[imageIndex] = image
				cloned[index].Images[imageIndex].Data = append([]byte(nil), image.Data...)
			}
		}
	}
	return cloned
}

func firstTaskLine(output string) string {
	if line, _, ok := strings.Cut(output, "\n"); ok {
		return truncateRunes(strings.TrimSpace(line), 120)
	}
	return truncateRunes(strings.TrimSpace(output), 120)
}

func send(runContext context.Context, events chan<- Event, event Event) {
	if events == nil {
		return
	}
	if event.Kind == Finished || event.Kind == Failed {
		events <- event
		return
	}
	select {
	case events <- event:
	case <-runContext.Done():
	}
}
