/*
上下文压缩：把较旧的完整回合变成有界事实检查点，保留最近回合的原始消息结构。
压缩只在用户消息边界切分，避免拆开近期工具调用与工具结果。
*/
package agent

import (
	"fmt"
	"strings"

	"github.com/momo-mnsjtxy/agentcore/model"
)

const compactSummaryLimit = 12_000

// CompactResult reports whether model context changed and by how much.
type CompactResult struct {
	BeforeMessages  int
	AfterMessages   int
	RemovedMessages int
}

// Compact replaces older turns with a bounded system checkpoint.
func (agent *Agent) Compact(keepUserTurns int) CompactResult {
	if keepUserTurns < 1 {
		keepUserTurns = 1
	}
	agent.mu.Lock()
	defer agent.mu.Unlock()

	before := len(agent.history) - 1
	cut := recentTurnStart(agent.history, keepUserTurns)
	if cut <= 1 {
		return CompactResult{BeforeMessages: before, AfterMessages: before}
	}
	checkpoint := summarizeMessages(agent.history[1:cut])
	recent := cloneMessages(agent.history[cut:])
	agent.history = append([]model.Message{
		agent.system,
		{Role: "system", Content: checkpoint},
	}, recent...)
	after := len(agent.history) - 1
	return CompactResult{
		BeforeMessages:  before,
		AfterMessages:   after,
		RemovedMessages: before - after,
	}
}

func recentTurnStart(messages []model.Message, keepUserTurns int) int {
	seen := 0
	for index := len(messages) - 1; index >= 1; index-- {
		if messages[index].Role != "user" {
			continue
		}
		seen++
		if seen == keepUserTurns {
			return index
		}
	}
	return 0
}

func summarizeMessages(messages []model.Message) string {
	toolNames := make(map[string]string)
	for _, message := range messages {
		if message.Role != "assistant" {
			continue
		}
		for _, call := range message.ToolCalls {
			toolNames[call.ID] = call.Function.Name
		}
	}

	var summary strings.Builder
	summary.WriteString("Conversation checkpoint from earlier completed turns. Treat this as factual history; preserve the recent messages that follow it.\n")
	for _, message := range messages {
		switch message.Role {
		case "system":
			appendSummary(&summary, "Earlier checkpoint", message.Content, 2_500)
		case "user":
			appendSummary(&summary, "User", message.Content, 1_200)
		case "assistant":
			appendSummary(&summary, "Assistant", message.Content, 1_800)
			for _, call := range message.ToolCalls {
				appendSummary(&summary, "Tool request "+call.Function.Name, call.Function.Arguments, 900)
			}
		case "tool":
			name := toolNames[message.ToolCallID]
			if name == "" {
				name = "unknown"
			}
			appendSummary(&summary, "Tool result "+name, message.Content, 1_400)
		}
		if summary.Len() >= compactSummaryLimit {
			break
		}
	}
	return truncateRunes(summary.String(), compactSummaryLimit)
}

func appendSummary(summary *strings.Builder, label, content string, limit int) {
	content = strings.TrimSpace(content)
	if content == "" || summary.Len() >= compactSummaryLimit {
		return
	}
	block := fmt.Sprintf("\n%s:\n%s\n", label, truncateRunes(content, limit))
	remaining := compactSummaryLimit - summary.Len()
	summary.WriteString(truncateRunes(block, remaining))
}

func truncateRunes(text string, limit int) string {
	if limit <= 0 {
		return ""
	}
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	if limit == 1 {
		return "…"
	}
	return string(runes[:limit-1]) + "…"
}
