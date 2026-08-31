/*
元工具：由核心自己消化、不交给 Toolbox 的模型工具。
干净核心只认识澄清——request_clarification 是循环停下来向用户要信息的通用能力。
工作流、计划、审查是产出机制，属于上层 agentengine，不在这里。
*/
package agent

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/momo-mnsjtxy/agentcore/model"
)

// metaTool 是一个由核心直接处理的模型工具：definition 是模型看到的合约，run 是核心侧的处理。
// run 返回 ok（这次调用是否成立）、回填给模型的文本，以及 stop（为真表示应立刻结束运行）。
type metaTool struct {
	definition model.Tool
	run        func(*Agent, context.Context, chan<- Event, <-chan ApprovalDecision, model.ToolCall) (bool, string, bool)
}

var metaTools = []metaTool{
	{
		run: (*Agent).requestClarification,
		definition: model.Tool{
			Type: "function",
			Function: model.FunctionTool{
				Name:        "request_clarification",
				Description: "Pause and ask the user for missing information before continuing.",
				Parameters: map[string]any{
					"type":     "object",
					"required": []string{"question"},
					"properties": map[string]any{
						"question": map[string]any{"type": "string"},
						"options":  map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
					},
				},
			},
		},
	},
}

// handleMeta 分派元工具。返回 handled 表示命中了元工具，stop 表示应立刻结束运行。
func (agent *Agent) handleMeta(runContext context.Context, events chan<- Event, approvals <-chan ApprovalDecision, call model.ToolCall) (bool, bool) {
	for _, meta := range metaTools {
		if call.Function.Name != meta.definition.Function.Name {
			continue
		}
		ok, output, stop := meta.run(agent, runContext, events, approvals, call)
		agent.appendMessage(model.Message{Role: "tool", ToolCallID: call.ID, Content: output})
		agent.recordToolOutcome(call, output, !ok)
		return true, stop
	}
	return false, false
}

func (agent *Agent) requestClarification(runContext context.Context, events chan<- Event, approvals <-chan ApprovalDecision, call model.ToolCall) (bool, string, bool) {
	var input struct {
		Question string   `json:"question"`
		Options  []string `json:"options"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &input); err != nil {
		return false, "Clarification request failed: " + err.Error(), false
	}
	input.Question = strings.TrimSpace(input.Question)
	if input.Question == "" {
		input.Question = "Please provide the missing information."
	}
	send(runContext, events, Event{Kind: Clarification, Text: input.Question, Call: call})

	decision, received := receiveDecision(runContext, approvals)
	if !received {
		send(runContext, events, Event{Kind: Finished, Text: "Cancelled"})
		return false, "", true
	}
	if !decision.Approved {
		return true, "User dismissed the clarification.", false
	}
	if answer := strings.TrimSpace(decision.Answer); answer != "" {
		return true, "User answered: " + answer, false
	}
	return true, "User answered the clarification.", false
}

func (agent *Agent) requestRepeatedFailureClarification(runContext context.Context, events chan<- Event, approvals <-chan ApprovalDecision, failedCall model.ToolCall, output string) bool {
	question := "The same action failed repeatedly: " + failedCall.Function.Name + ". How should I proceed?"
	arguments, _ := json.Marshal(map[string]any{
		"question": question,
		"options":  []string{"Retry with a different approach", "Stop and explain the blocker", "Show the full output"},
	})
	call := model.ToolCall{
		ID:       "clarify_repeated_failure",
		Type:     "function",
		Function: model.FunctionCall{Name: "request_clarification", Arguments: string(arguments)},
	}
	agent.appendMessage(model.Message{Role: "assistant", ToolCalls: []model.ToolCall{call}})
	send(runContext, events, Event{Kind: Clarification, Text: question, Call: call})
	answer := "User dismissed the clarification."
	decision, received := receiveDecision(runContext, approvals)
	if !received {
		send(runContext, events, Event{Kind: Finished, Text: "Cancelled"})
		return true
	}
	if decision.Approved && strings.TrimSpace(decision.Answer) != "" {
		answer = "User answered: " + strings.TrimSpace(decision.Answer)
	}
	agent.appendMessage(model.Message{Role: "tool", ToolCallID: call.ID, Content: answer})
	agent.recordToolOutcome(call, answer, false)
	return false
}

func receiveDecision(runContext context.Context, approvals <-chan ApprovalDecision) (ApprovalDecision, bool) {
	if approvals == nil {
		return ApprovalDecision{}, true
	}
	select {
	case decision, open := <-approvals:
		if !open {
			return ApprovalDecision{}, true
		}
		return decision, true
	case <-runContext.Done():
		return ApprovalDecision{}, false
	}
}
