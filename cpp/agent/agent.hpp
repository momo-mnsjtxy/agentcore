/*
Agent 认知核心：把用户输入交给模型，执行模型选择的工具，再把观察结果送回模型。
公开入口只有 Agent::run；审批、澄清、快照和压缩都围绕同一条认知循环工作。
*/
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../model/chat.hpp"

namespace agentcore::agent {

// EventKind 描述认知循环当前发生的业务动作。
enum class EventKind {
    delta,
    reasoning,
    messageDone,
    approval,
    toolStarted,
    toolDone,
    usageChanged,
    clarification,
    finished,
    failed,
};

// eventKindName 把事件转成稳定的界面协议名称。
std::string eventKindName(EventKind kind);

// Event 是认知循环交给界面的一次变化。
struct Event {
    EventKind kind;
    std::string text;
    model::ToolCall call;
    model::Usage usage;
    bool failed = false;

    Event(EventKind kindValue,
          std::string textValue = {},
          model::ToolCall callValue = {},
          model::Usage usageValue = {},
          bool failedValue = false)
        : kind(kindValue),
          text(std::move(textValue)),
          call(std::move(callValue)),
          usage(usageValue),
          failed(failedValue) {}
};

// ApprovalDecision 是用户对高风险工具动作或澄清请求的决定。
struct ApprovalDecision {
    bool approved = false;
    bool remember = false;
    std::string answer;
};

// Stats 是当前会话的轻量运行数据，不暴露消息内容。
struct Stats {
    int turns = 0;
    int messages = 0;
    int tools = 0;
    int approvals = 0;
    int failures = 0;
};

// Snapshot 是可持久化的会话内容，不包含系统提示和审批许可。
struct Snapshot {
    std::vector<model::Message> messages;
    int turns = 0;
};

// Capability 描述工具对环境的改动程度。
enum class Capability {
    read,
    write,
    command,
};

// Toolbox 是 Agent 可以观察和改变环境的能力集合。
class Toolbox {
public:
    virtual ~Toolbox() = default;

    virtual std::vector<model::Tool> definitions() const = 0;
    virtual Capability capability(const model::ToolCall& call) const = 0;
    virtual bool needsApproval(const model::ToolCall& call) const = 0;
    virtual std::string preview(const model::ToolCall& call) const = 0;
    virtual model::Observation execute(model::Context& context,
                                        const model::ToolCall& call) = 0;
};

// WorkspaceDiffer 让调用方按需提供完整环境差异。
class WorkspaceDiffer {
public:
    virtual ~WorkspaceDiffer() = default;
    virtual std::string diff(model::Context& context) = 0;
};

// WorkspaceUndoer 让调用方回退最近一次结构化改动。
class WorkspaceUndoer {
public:
    virtual ~WorkspaceUndoer() = default;
    virtual std::string undo() = 0;
};

// EventSink 接收界面要显示的事件；空 sink 表示调用方只关心最终状态。
using EventSink = std::function<void(const Event&)>;

// ApprovalSource 同步提供一次用户决定；空 source 等同立即拒绝。
using ApprovalSource =
    std::function<std::optional<ApprovalDecision>(model::Context&)>;

// CompactResult 报告上下文压缩前后的消息数量。
struct CompactResult {
    int beforeMessages = 0;
    int afterMessages = 0;
    int removedMessages = 0;
};

// Agent 持有会话状态和完整认知循环。
class Agent {
public:
    Agent(std::shared_ptr<model::Provider> provider,
          std::shared_ptr<Toolbox> tools,
          std::string system = {});

    // defaultSystem 返回应用可以叠加产品身份的通用行为提示。
    static std::string defaultSystem();

    // reset 清空会话并保留同一套系统提示。
    void reset();

    // stats 返回当前会话的计数，不复制消息正文。
    Stats stats() const;

    // snapshot 返回脱离 Agent 的会话副本，调用方可自行保存。
    Snapshot snapshot() const;

    // restore 替换会话并清除本进程之前记住的审批。
    void restore(const Snapshot& snapshot);

    // run 追踪一次用户输入，直到模型完成、失败或 context 被取消。
    void run(model::Context& context,
             std::string_view input,
             const EventSink& events = {},
             const ApprovalSource& approvals = {});

    // compact 把较旧回合压成有界事实检查点，保留最近的用户回合。
    CompactResult compact(int keepUserTurns);

    // workspaceDiff 在 Toolbox 支持时返回完整可检查的环境差异。
    std::string workspaceDiff(model::Context& context);

    // undoLastChange 在 Toolbox 支持时回退最近一次结构化改动。
    std::string undoLastChange();

private:
    friend class Meta;

    std::vector<model::Tool> toolDefinitions() const;
    std::vector<model::Message> startTurn();
    void appendMessage(const model::Message& message);
    void rememberApproval(const std::string& key);
    bool isApproved(const std::string& key) const;
    void countApproval();
    void countToolRun(bool failed);
    bool recordToolOutcome(const model::ToolCall& call,
                           const std::string& output,
                           bool failed);
    void refuseCall(model::Context& context,
                    const EventSink& events,
                    const model::ToolCall& call,
                    std::string reason);
    bool handleMeta(model::Context& context,
                    const EventSink& events,
                    const ApprovalSource& approvals,
                    const model::ToolCall& call,
                    bool& stop);
    bool requestClarification(model::Context& context,
                              const EventSink& events,
                              const ApprovalSource& approvals,
                              const model::ToolCall& call,
                              std::string& output,
                              bool& stop);
    bool requestRepeatedFailureClarification(
        model::Context& context,
        const EventSink& events,
        const ApprovalSource& approvals,
        const model::ToolCall& failedCall,
        const std::string& output);

    std::shared_ptr<model::Provider> provider_;
    std::shared_ptr<Toolbox> tools_;
    model::Message system_;
    mutable std::mutex mutex_;
    std::vector<model::Message> history_;
    std::unordered_map<std::string, bool> approved_;
    int turns_ = 0;
    int toolRuns_ = 0;
    int approvals_ = 0;
    int toolFailures_ = 0;
    std::string failureKey_;
    int failureStreak_ = 0;
};

}  // namespace agentcore::agent
