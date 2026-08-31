/*
模型对话协议：描述 Agent、模型服务与工具之间交换的数据。
业务循环只依赖这些简单结构，模型供应商可以独立替换。
*/
package model

import "context"

// Message 是一条模型可理解的对话消息。
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	Images     []Image    `json:"-"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Image 是送给模型看的一张图。字节不写入会话文件，避免检查点膨胀。
type Image struct {
	MIMEType string
	Data     []byte
}

// Observation 是一次工具执行后模型应看到的结果。
type Observation struct {
	Text   string
	Images []Image
}

// ToolCall 是模型发起的一次工具调用。
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall 保存工具名与模型生成的 JSON 参数。
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool 描述模型可以调用的一个本地能力。
type Tool struct {
	Type     string       `json:"type"`
	Function FunctionTool `json:"function"`
}

// FunctionTool 是函数工具的名称、说明与参数结构。
type FunctionTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// Reply 是模型一轮输出的完整结果。
type Reply struct {
	Content    string
	ToolCalls  []ToolCall
	Usage      Usage
	ResponseID string
}

// Usage 是当前会话最近一次已知的 token 用量。
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// StreamKind 区分回答、推理摘要和用量变化。
type StreamKind string

const (
	OutputDelta    StreamKind = "output_delta"
	ReasoningDelta StreamKind = "reasoning_delta"
	UsageChanged   StreamKind = "usage_changed"
)

// StreamEvent 是模型协议向 Agent 发出的一个增量事件。
type StreamEvent struct {
	Kind  StreamKind
	Text  string
	Usage Usage
}

// Stream 接收模型增量事件。
type Stream func(StreamEvent)

// Provider 完成一轮流式推理。
type Provider interface {
	Complete(context.Context, []Message, []Tool, Stream) (Reply, error)
}
