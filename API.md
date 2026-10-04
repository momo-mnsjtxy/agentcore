# Agentcore 接口文档

本文描述仓库当前的 Go 和 C++20 公共接口。Agentcore 是嵌入应用的认知核心库，调用方提供模型、工具、系统提示和交互界面。

Go 模块路径：`github.com/momo-mnsjtxy/agentcore`，当前 `go.mod` 要求 Go 1.26.4。C++ 目标名为 `agentcore_cpp`，依赖 C++20、libcurl 和 JsonCpp。许可证为 [AGPL-3.0-only](LICENSE)。

## 目录

- [接入示例](#接入示例)
- [模型数据与 Provider](#模型数据与-provider)
- [Go Agent](#go-agent)
- [工具与可选扩展](#工具与可选扩展)
- [事件、审批与澄清](#事件审批与澄清)
- [会话快照与压缩](#会话快照与压缩)
- [OpenAI 兼容客户端](#openai-兼容客户端)
- [C++ 接口](#c-接口)
- [接入约束](#接入约束)

## 接入示例

下面是完整的 Go 接入程序：暴露一个只读 `echo` 工具，持续消费事件，并在运行结束后读取会话状态。审批默认拒绝，澄清回复由应用界面提供后再通过 `Answer` 传入。

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "os"
    "time"

    "github.com/momo-mnsjtxy/agentcore/agent"
    "github.com/momo-mnsjtxy/agentcore/model"
    "github.com/momo-mnsjtxy/agentcore/openai"
)

type Toolbox struct{}

func (Toolbox) Definitions() []model.Tool {
    return []model.Tool{{
        Type: "function",
        Function: model.FunctionTool{
            Name: "echo", Description: "返回输入文本。",
            Parameters: map[string]any{
                "type": "object",
                "properties": map[string]any{
                    "text": map[string]any{"type": "string"},
                },
                "required": []string{"text"},
                "additionalProperties": false,
            },
        },
    }}
}

func (Toolbox) Capability(model.ToolCall) agent.Capability {
    return agent.CapabilityRead
}
func (Toolbox) NeedsApproval(model.ToolCall) bool { return false }
func (Toolbox) Preview(call model.ToolCall) string {
    return call.Function.Name + " " + call.Function.Arguments
}
func (Toolbox) Execute(ctx context.Context, call model.ToolCall) (model.Observation, error) {
    if err := ctx.Err(); err != nil {
        return model.Observation{}, err
    }
    if call.Function.Name != "echo" {
        return model.Observation{}, fmt.Errorf("unknown tool: %s", call.Function.Name)
    }
    var input struct { Text string `json:"text"` }
    if err := json.Unmarshal([]byte(call.Function.Arguments), &input); err != nil {
        return model.Observation{}, err
    }
    return model.Observation{Text: input.Text}, nil
}

func main() {
    // BASE_URL 包含服务的 API 前缀，例如 https://api.example.com/v1。
    provider := openai.New(os.Getenv("BASE_URL"), os.Getenv("API_KEY"),
        os.Getenv("MODEL"), "responses", "none")
    brain := agent.New(provider, Toolbox{}, agent.DefaultSystem())
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    events := make(chan agent.Event, 32)
    decisions := make(chan agent.ApprovalDecision, 1)
    done := make(chan struct{})

    go func() {
        defer close(done)
        defer close(events) // 核心不关闭调用方的 channel。
        brain.RunWithOptions(ctx, "用 echo 返回 hello", agent.RunOptions{
            MaxTurns: 8, MaxTools: 4, Timeout: 2 * time.Minute,
        }, events, decisions)
    }()

    for event := range events {
        switch event.Kind {
        case agent.Delta:
            fmt.Print(event.Text)
        case agent.Approval, agent.Clarification:
            fmt.Println("\n等待用户决定：", event.Text)
            decisions <- agent.ApprovalDecision{Approved: false}
        case agent.Failed, agent.Paused:
            fmt.Println("\n", event.Text)
        }
    }
    <-done
    fmt.Printf("\n会话状态：%s\n", brain.Stats().Status)
}
```

## 模型数据与 Provider

Go 包：`model`；C++ 命名空间：`agentcore::model`，头文件：`cpp/model/chat.hpp`。

### 公共数据

| Go 类型 | 字段 | 含义 |
| --- | --- | --- |
| `Message` | `Role string`、`Content string`、`Images []Image`、`ToolCalls []ToolCall`、`ToolCallID string` | 会话消息；工具结果使用 `Role: "tool"`，并通过 `ToolCallID` 对应调用 |
| `Image` | `MIMEType string`、`Data []byte` | 图片 MIME 类型和原始字节，如 `image/png` |
| `Observation` | `Text string`、`Images []Image` | 工具执行结果 |
| `ToolCall` | `ID string`、`Type string`、`Function FunctionCall` | 一次模型工具调用，`Type` 通常为 `function` |
| `FunctionCall` | `Name string`、`Arguments string` | 工具名与 JSON 参数字符串；参数由工具执行器解析 |
| `Tool` | `Type string`、`Function FunctionTool` | 模型可见的工具定义 |
| `FunctionTool` | `Name string`、`Description string`、`Parameters map[string]any` | 工具说明和参数 JSON Schema |
| `Reply` | `Content string`、`ToolCalls []ToolCall`、`Usage Usage`、`ResponseID string` | 单轮模型请求的完整结果 |
| `Usage` | `InputTokens int`、`OutputTokens int`、`TotalTokens int` | 单次模型回答报告的 token 用量 |
| `StreamEvent` | `Kind StreamKind`、`Text string`、`Usage Usage` | 模型增量事件 |

`StreamKind` 的取值是 `OutputDelta = "output_delta"`、`ReasoningDelta = "reasoning_delta"`、`UsageChanged = "usage_changed"`。`Stream` 的类型是 `func(StreamEvent)`。

### Provider 合约

```go
type Provider interface {
    Complete(context.Context, []Message, []Tool, Stream) (Reply, error)
}
```

`Complete` 完成一轮推理，向 `Stream` 发送增量，并返回完整回答。它只生成工具调用，实际执行由 Agent 和 Toolbox 完成。自定义实现应处理取消、允许空流回调，并让最终 `Reply` 包含本轮完整文本、工具调用和用量。

C++ 对应接口：

```cpp
virtual Reply complete(Context& context,
                       const std::vector<Message>& messages,
                       const std::vector<Tool>& tools,
                       const Stream& stream) = 0;
```

C++ 字段名使用小驼峰，例如 `toolCalls`、`toolCallID`、`mimeType`、`inputTokens`。容器分别为 `std::vector`、图片字节为 `std::vector<std::uint8_t>`、工具参数 Schema 为 `Json::Value`。

## Go Agent

Go 包：`agent`。

### 创建与运行

```go
func DefaultSystem() string
func New(provider model.Provider, tools Toolbox, system string) *Agent

func (a *Agent) Run(ctx context.Context, input string,
    events chan<- Event, approvals <-chan ApprovalDecision)
func (a *Agent) RunWithOptions(ctx context.Context, input string, options RunOptions,
    events chan<- Event, approvals <-chan ApprovalDecision)
func (a *Agent) Resume(ctx context.Context,
    events chan<- Event, approvals <-chan ApprovalDecision)
func (a *Agent) ResumeWithOptions(ctx context.Context, options RunOptions,
    events chan<- Event, approvals <-chan ApprovalDecision)
```

- `New` 要求有效的 Provider 和 Toolbox；空白系统提示使用 `DefaultSystem()`。提供自定义提示时，核心不会自动叠加默认提示。
- `Run` 和 `RunWithOptions` 追加用户输入并持续运行工具闭环。输入会去除首尾空白，空输入直接发送 `finished`。
- `Run` 每次最多执行 32 轮模型推理，耗尽轮次发送 `failed`。
- `RunWithOptions` 在预算耗尽时设置 `paused` 状态。运行预算按本次方法调用重新计算。
- `Resume` 和 `ResumeWithOptions` 继续快照中的 `Goal`，不重复追加用户输入。没有目标时发送 `failed`。
- 方法同步执行、无返回值；结果通过事件和 `Stats()` 读取。同一 Go Agent 同时只能运行一个循环，重复运行请求发送 `failed`。

### 运行预算

```go
type RunOptions struct {
    MaxTurns  int
    MaxTools  int
    MaxTokens int
    Timeout   time.Duration
}
```

| 字段 | 默认与检查方式 |
| --- | --- |
| `MaxTurns` | 小于等于 0 时使用 32；每次 Provider 请求计一轮 |
| `MaxTools` | 小于等于 0 不限制；执行一批已批准的普通工具前检查，整批会超限时暂停；澄清元工具不计入 |
| `MaxTokens` | 小于等于 0 不限制；累计每轮 `Reply.Usage.TotalTokens`，模型回答后检查，因此可能超过设定值；服务未报告的用量无法计入 |
| `Timeout` | 大于 0 时为本次运行创建超时 context；调用方及 Provider、Toolbox 必须配合取消 |

预算暂停可能发生在 assistant 工具调用已经写入历史、工具结果尚未回填时。当前 `Resume*` 会再次调用 Provider，并不直接重放这批未执行调用；恢复能力需要结合所用服务的消息协议验证。

### 状态与统计

```go
func (a *Agent) Stats() Stats
func (a *Agent) Reset()
```

`Stats` 包含 `Turns`（模型请求累计次数）、`Messages`（不含初始系统提示的消息数）、`Tools`（普通工具实际执行次数）、`Approvals`（发出的审批请求数）、`Failures`（普通工具执行失败次数）、`Goal` 和 `Status`。

`RunStatus`：`idle`、`running`、`paused`、`finished`、`failed`，对应常量 `StatusIdle`、`StatusRunning`、`StatusPaused`、`StatusFinished`、`StatusFailed`。

`Reset` 清空历史、审批记忆、统计和目标，保留构造时的系统提示。取消后的状态可能为 `failed`，即使终止事件是 `finished` 且文本为 `Cancelled`；调用方应结合事件文本和 context 判断取消。

## 工具与可选扩展

### Toolbox：五个必需方法

```go
type Toolbox interface {
    Definitions() []model.Tool
    Capability(model.ToolCall) Capability
    NeedsApproval(model.ToolCall) bool
    Preview(model.ToolCall) string
    Execute(context.Context, model.ToolCall) (model.Observation, error)
}
```

| 方法 | 调用方责任 |
| --- | --- |
| `Definitions` | 提供模型可见的函数定义和参数 Schema |
| `Capability` | 描述工具能力：`CapabilityRead`、`CapabilityWrite`、`CapabilityCommand` |
| `NeedsApproval` | 决定该调用是否需要用户批准；当前核心实际以此方法控制审批，并不调用 `Capability` 判定 |
| `Preview` | 提供用户可检查的操作说明；结果也参与审批记忆键 |
| `Execute` | 解析参数、执行动作并返回文字或图片；通过 error 表示执行失败 |

工具失败时，核心将错误追加到观察文本，写入会话并继续交给模型处理。拒绝执行也会产生 `tool_done`，但 `Failed` 为 `false`，并回填 `User denied this action.`。

### 并行执行

```go
type ToolConcurrency interface {
    ParallelSafe(model.ToolCall) bool
}
```

默认串行。仅当 Toolbox 实现此接口、本轮待执行工具超过一个，并且每个调用都返回 `true` 时，整批工具并行运行。审批仍顺序处理；执行完成后按模型调用顺序写入结果。Toolbox 应保证声明为安全的调用及其共享状态支持并发。

### 完成验证

```go
type CompletionVerifier interface {
    Verify(context.Context, string, []model.Message) (VerificationResult, error)
}
type VerificationResult struct {
    Complete  bool
    Evidence  string
    Feedback  string
    Retryable bool
}
```

模型返回无工具调用的回答时，核心调用 `Verify`；第二个参数是该轮完整文本，第三个参数是当前消息快照。

- `Complete: true` 且无 error：运行完成。
- 未通过且 `Retryable: true`：将反馈写入历史，继续推理。
- 未通过且 `Retryable: false`：进入 `paused`，等待调用方介入。这个行为也适用于普通 `Run`。
- error 表示验证失败；是否继续仍由结果的 `Retryable` 决定。

反馈优先使用 `Feedback`，为空时使用 `Evidence`，二者都为空时使用核心默认文本。应用应检查实际测试、文件或业务状态。未实现此接口时，无工具调用的回答直接视为完成。

### 环境差异与撤销

```go
type WorkspaceDiffer interface { Diff(context.Context) (string, error) }
type WorkspaceUndoer interface { Undo() (string, error) }

func (a *Agent) WorkspaceDiff(ctx context.Context) (string, error)
func (a *Agent) UndoLastChange() (string, error)
```

Agent 将请求转交给实现这些接口的 Toolbox。未实现时分别返回 `workspace diff is unavailable` 或 `workspace undo is unavailable`。核心不记录文件变更，也不自行执行撤销。

## 事件、审批与澄清

```go
type Event struct {
    Kind   EventKind
    Text   string
    Call   model.ToolCall
    Usage  model.Usage
    Failed bool
}
type ApprovalDecision struct {
    Approved bool
    Remember bool
    Answer   string
}
```

| Go 常量 / 事件值 | 含义与主要字段 |
| --- | --- |
| `Delta` / `delta` | 回答文本增量，读取 `Text` |
| `Reasoning` / `reasoning` | 推理摘要增量，读取 `Text` |
| `MessageDone` / `message_done` | 一轮模型回答结束，不等于整个运行结束；文本已通过增量发送 |
| `Approval` / `approval` | 等待用户批准，`Text` 为预览、`Call` 为调用 |
| `ToolStarted` / `tool_started` | 工具开始执行，携带预览和调用 |
| `ToolDone` / `tool_done` | 工具结果或拒绝说明，读取 `Text`、`Call`、`Failed` |
| `UsageChanged` / `usage_changed` | 本轮用量；可能由流和最终回答重复发送，不应直接对所有事件求和 |
| `Clarification` / `clarification` | 请求用户补充信息，`Text` 为问题、选项位于 `Call.Function.Arguments` |
| `Paused` / `paused` | 预算耗尽或完成验证需要介入，`Text` 为原因；Go 专有 |
| `Finished` / `finished` | 正常结束或取消；取消时 `Text` 通常为 `Cancelled` |
| `Failed` / `failed` | 模型错误、运行冲突或轮次耗尽等，读取 `Text` |

### 审批反馈

收到 `approval` 后，通过同一个 decisions channel 发送一次决定：

```go
decisions <- agent.ApprovalDecision{Approved: true, Remember: true}
```

`Remember` 只在批准时生效。记忆键是工具名和 `Preview` 的组合，并非对该工具的所有调用授权；`Reset` 和 `Restore` 会清除它。`approvals == nil` 或 channel 已关闭时立即拒绝，无需等待。

### 澄清合约

核心识别 `request_clarification`，参数如下：

```json
{
  "question": "要处理哪个目录？",
  "options": ["src", "tests"]
}
```

参数 Schema 为 object，`question` 是必需的 string，`options` 是可选的 string 数组。收到 `clarification` 后，发送：

```go
decisions <- agent.ApprovalDecision{Approved: true, Answer: "src"}
```

Go 当前的 `Definitions()` 不自动合并这个元工具，调用方需将上述合约加入自己的工具列表，才能让模型主动请求澄清；核心会拦截这个名字，不交给 `Execute`。C++ 会自动合并，调用方无需重复定义。

同一工具的相同错误首行连续出现三次时，核心也会发起澄清。这里等待回答后继续循环，并非持久化的预算 `paused` 状态。`Approved: false` 表示用户关闭澄清请求，核心回填关闭说明后继续。

## 会话快照与压缩

```go
type Snapshot struct {
    Messages []model.Message `json:"messages"`
    Turns    int             `json:"turns"`
    Goal     string          `json:"goal,omitempty"`
    Status   RunStatus       `json:"status,omitempty"`
}
func (a *Agent) Snapshot() Snapshot
func (a *Agent) Restore(snapshot Snapshot)

type CompactResult struct {
    BeforeMessages  int
    AfterMessages   int
    RemovedMessages int
}
func (a *Agent) Compact(keepUserTurns int) CompactResult
```

`Snapshot` 返回会话副本，不含构造时的系统提示和审批记忆。`Restore` 使用目标 Agent 的系统提示，恢复消息、模型轮次、目标和状态，并清零工具、审批、失败统计。

快照的序列化、存储、原子写入和列表管理均由调用方实现。Go 的 `Message.Images` 带有 `json:"-"`，标准 JSON 编码不会保存图片；如需恢复图像内容，应用应单独持久化。

序列化与恢复的调用片段：

```go
data, err := json.Marshal(brain.Snapshot())
if err != nil { return err }
// 由调用方将 data 持久化，并在需要时读回。
var snapshot agent.Snapshot
if err := json.Unmarshal(data, &snapshot); err != nil { return err }
brain.Restore(snapshot)
brain.ResumeWithOptions(ctx, agent.RunOptions{MaxTurns: 16}, events, decisions)
```

`Compact` 保留最近 `keepUserTurns` 个用户回合，在用户消息边界压缩较旧内容，避免拆开近期工具调用和结果。小于 1 时按 1 处理；无需压缩时返回原消息数。压缩使用本地文本截断和汇总，不请求模型，摘要受 12,000 上限约束。

两种实现都只保留最近三个**含图片的消息**中的图片；一条消息内可以有多张图片。

## OpenAI 兼容客户端

Go 包：`openai`。

```go
func New(baseURL, apiKey, modelName, protocol, reasoning string) *Client
func (c *Client) Complete(ctx context.Context, messages []model.Message,
    tools []model.Tool, stream model.Stream) (model.Reply, error)
```

| 参数 | 行为 |
| --- | --- |
| `baseURL` | API 前缀，避免末尾 `/`；客户端直接拼接路径 |
| `apiKey` | 非空时设置 `Authorization: Bearer ...` |
| `modelName` | 服务支持的模型标识 |
| `protocol` | 仅精确等于 `responses` 时请求 `/responses`，其他值包括空字符串都请求 `/chat/completions` |
| `reasoning` | Responses 路径中，非空且不为 `none` 时发送 `reasoning.effort` 和 `summary: "auto"`；Chat 路径不使用 |

协议选择在构造时固定；Responses 出错不会自动再试 Chat。请求使用 SSE；Responses 发送 `store: false` 和 `parallel_tool_calls: false`，每轮显式携带当前会话。

Go 客户端最多尝试三次。部分连接错误及 HTTP 429、500、502、503、504 可以重试；已经产生有效流输出后不重新请求，避免重复输出。Go 客户端使用环境代理；应用可通过运行 context 控制请求的取消和总时限。

## C++ 接口

### Agent 与 Toolbox

头文件：`cpp/agent/agent.hpp`，命名空间：`agentcore::agent`。

```cpp
Agent(std::shared_ptr<model::Provider> provider,
      std::shared_ptr<Toolbox> tools,
      std::string system = {});

static std::string defaultSystem();
void reset();
Stats stats() const;
Snapshot snapshot() const;
void restore(const Snapshot& snapshot);
void run(model::Context& context, std::string_view input,
         const EventSink& events = {}, const ApprovalSource& approvals = {});
CompactResult compact(int keepUserTurns);
std::string workspaceDiff(model::Context& context);
std::string undoLastChange();
```

Provider 和 Toolbox 由 `shared_ptr` 持有；空指针使构造函数抛出 `std::invalid_argument`。Toolbox 的五个纯虚方法：

```cpp
std::vector<model::Tool> definitions() const;
Capability capability(const model::ToolCall& call) const;
bool needsApproval(const model::ToolCall& call) const;
std::string preview(const model::ToolCall& call) const;
model::Observation execute(model::Context& context, const model::ToolCall& call);
```

`execute` 抛出 `std::exception` 表示工具失败；核心将失败转换为工具结果。可选的 `WorkspaceDiffer::diff(Context&)` 和 `WorkspaceUndoer::undo()` 返回字符串，未实现时 Agent 对应入口抛出异常。

### 回调与取消

```cpp
using EventSink = std::function<void(const Event&)>;
using ApprovalSource =
    std::function<std::optional<ApprovalDecision>(model::Context&)>;
```

事件通过同步回调发送。`EventKind` 采用小驼峰枚举名，例如 `messageDone`、`toolStarted`；`eventKindName(EventKind)` 返回与 Go 相同的字符串事件名。C++ 不提供 `paused` 事件。

`ApprovalDecision` 字段为 `approved`、`remember`、`answer`。空审批回调立即拒绝；回调返回 `std::nullopt` 表示停止当前等待并结束运行。回调应自行接入界面并响应取消。

`model::Context` 提供 `cancel()`、`cancelled()` 和 `throwIfCancelled()`。取消后不能重置，应为下一次运行新建 Context；`throwIfCancelled()` 抛出 `model::Cancelled`。

使用应用实现的 `provider` 和 `tools`：

```cpp
agentcore::agent::Agent brain(provider, tools);
agentcore::model::Context context;
brain.run(context, "检查项目", [](const agentcore::agent::Event& event) {
    if (event.kind == agentcore::agent::EventKind::delta) {
        std::cout << event.text;
    }
}, [](agentcore::model::Context& context)
       -> std::optional<agentcore::agent::ApprovalDecision> {
    if (context.cancelled()) return std::nullopt;
    return agentcore::agent::ApprovalDecision{false, false, {}};
});
```

### C++ 客户端与协议辅助函数

头文件：`cpp/openai/openai.hpp`，命名空间：`agentcore::openai`。

```cpp
Client(std::string baseURL, std::string apiKey, std::string modelName,
       std::string protocol = "responses", std::string reasoning = "none");

model::Reply complete(model::Context& context,
                      const std::vector<model::Message>& messages,
                      const std::vector<model::Tool>& tools,
                      const model::Stream& stream) override;

ParsedReply readChatStream(std::string_view source, const model::Stream& stream = {});
ParsedReply readResponsesStream(std::string_view source, const model::Stream& stream = {});
Json::Value chatPayload(const std::string& modelName,
                        const std::vector<model::Message>& messages,
                        const std::vector<model::Tool>& tools);
Json::Value responsesPayload(const std::string& modelName, const std::string& reasoning,
                             const std::vector<model::Message>& messages,
                             const std::vector<model::Tool>& tools);
std::string dataURL(const model::Image& image);
```

`ParsedReply` 包含 `model::Reply reply` 和 `bool progressed`；后者表示解析中是否已出现有效输出，用于重试判断。解析函数接收 SSE 文本，编码函数返回可检查的请求 JSON。`model::toJson` 为 `FunctionCall`、`ToolCall`、`Tool` 和 `Usage` 提供 JSON 编码；Message 和 Snapshot 的持久化编码由应用实现。

### 两种实现的差异

| 能力 | Go | C++ |
| --- | --- | --- |
| 认知循环、审批、澄清、快照、压缩、图片观察 | 支持 | 支持 |
| 单次默认模型轮次上限 | 32 | 32 |
| 预算与恢复入口 | `RunWithOptions`、`Resume*` | 暂未提供 |
| 生命周期状态和目标 | `Stats`、`Snapshot` 含 `Goal/Status` | 统计和快照无目标、状态字段 |
| 完成验证、工具并行扩展 | 支持 | 暂未提供，工具串行执行 |
| 元工具定义 | 调用方加入 `request_clarification` | 核心自动合并 |
| 错误表达 | error 与事件 | 异常与事件 |
| 交互传输 | channel | 同步回调 |

## 接入约束

- 持续消费 Go 事件直到运行方法返回。终止事件的发送不受取消打断，停止消费会使运行阻塞；不要仅凭 context 已取消就退出事件消费者。
- channel 由调用方创建和关闭。运行结束前不要关闭 events；审批和澄清共用 decisions channel，按收到的请求逐个响应。
- 一次运行期间避免调用 `Reset`、`Restore`、`Compact` 修改会话。C++ 同一 Agent 的 `run` 也应由应用保证串行调用。
- 工具定义的 JSON Schema 是模型输入合约；Toolbox 执行器仍应校验工具名和参数，并自行实现权限边界及取消处理。
- `Resume*` 恢复的是认知上下文和目标，外部副作用、幂等性、事务及持久化一致性由应用管理。

实现依据：[Go Agent](agent/agent.go)、[Go 模型协议](model/chat.go)、[Go 客户端](openai/openai.go)、[C++ Agent](cpp/agent/agent.hpp)、[C++ 模型协议](cpp/model/chat.hpp)、[C++ 客户端](cpp/openai/openai.hpp)。
