/*
Agent 主循环：接收用户目标、让模型推理、顺序执行工具并回填观察结果。
每轮只沿“推理 → 工具 → 观察 → 继续”前进，直到完成、取消或遇到明确阻塞。
*/
#include "agent.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace agentcore::agent {
namespace {

constexpr int maxTurns = 32;              // 限制一次输入消耗的模型轮数
constexpr int maxStalledTurns = 3;        // 连续无业务进展时暂停

std::string trim(std::string_view text) {
    std::size_t first = 0;
    while (first < text.size() &&
           std::isspace(static_cast<unsigned char>(text[first]))) {
        ++first;
    }
    std::size_t last = text.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(text[last - 1]))) {
        --last;
    }
    return std::string(text.substr(first, last - first));
}

void emit(model::Context& context, const EventSink& events, Event event) {
    if (!events) {
        return;  // 调用方没有界面时仍然可以运行完整认知循环
    }
    if (event.kind != EventKind::finished &&
        event.kind != EventKind::failed && context.cancelled()) {
        return;  // 取消后不再把过期的中间状态送给界面
    }
    events(event);
}

}  // namespace

std::string eventKindName(EventKind kind) {
    switch (kind) {
    case EventKind::delta:
        return "delta";
    case EventKind::reasoning:
        return "reasoning";
    case EventKind::messageDone:
        return "message_done";
    case EventKind::approval:
        return "approval";
    case EventKind::toolStarted:
        return "tool_started";
    case EventKind::toolDone:
        return "tool_done";
    case EventKind::usageChanged:
        return "usage_changed";
    case EventKind::clarification:
        return "clarification";
    case EventKind::finished:
        return "finished";
    case EventKind::failed:
        return "failed";
    }
    return "failed";  // 未知枚举不能静默变成成功状态
}

Agent::Agent(std::shared_ptr<model::Provider> provider,
             std::shared_ptr<Toolbox> tools,
             std::string system)
    : provider_(std::move(provider)), tools_(std::move(tools)) {
    if (!provider_ || !tools_) {
        throw std::invalid_argument("Agent needs a provider and a toolbox");
    }
    if (trim(system).empty()) {
        system = defaultSystem();
    }
    system_ = model::Message{"system", std::move(system)};
    history_.push_back(system_);  // 系统提示是每次推理的第一条约束
}

std::string Agent::defaultSystem() {
    return
        "Complete the user's goal end to end. Inspect relevant files before "
        "editing. Keep the main business flow obvious, make the smallest "
        "coherent change, and verify important behavior after edits. Use "
        "tools whenever evidence is available locally. Continue through tool "
        "results until the task is genuinely complete. Explain blockers plainly "
        "instead of pretending success.\n\n"
        "Read tools are automatic. Commands and file changes require user "
        "approval. Never claim an action happened unless its tool result "
        "confirms it.";
}

void Agent::run(model::Context& context,
                std::string_view input,
                const EventSink& events,
                const ApprovalSource& approvals) {
    std::string goal = trim(input);
    if (goal.empty()) {
        emit(context, events, Event{EventKind::finished});
        return;  // 空输入不应污染会话或消耗模型调用
    }

    appendMessage(model::Message{"user", std::move(goal)});
    int stalledTurns = 0;

    for (int turn = 0; turn < maxTurns; ++turn) {
        if (context.cancelled()) {
            emit(context, events, Event{EventKind::finished, "Cancelled"});
            return;  // 用户已经取消时不再发起下一轮模型请求
        }
        std::vector<model::Message> messages = startTurn();
        model::Reply reply;
        try {
            reply = provider_->complete(
                context, messages, toolDefinitions(),
                [&](const model::StreamEvent& streamEvent) {
                    if (streamEvent.kind == model::StreamKind::outputDelta) {
                        emit(context, events,
                             Event{EventKind::delta, streamEvent.text});
                    } else if (streamEvent.kind ==
                               model::StreamKind::reasoningDelta) {
                        emit(context, events,
                             Event{EventKind::reasoning, streamEvent.text});
                    } else if (streamEvent.kind ==
                               model::StreamKind::usageChanged) {
                        emit(context, events,
                             Event{EventKind::usageChanged, {}, {},
                                  streamEvent.usage});
                    }
                });
        } catch (const model::Cancelled&) {
            emit(context, events,
                 Event{EventKind::finished, "Cancelled"});
            return;
        } catch (const std::exception& failure) {
            emit(context, events,
                 Event{EventKind::failed, failure.what(), {}, {}, true});
            return;
        }

        appendMessage(
            model::Message{"assistant", reply.content, {}, reply.toolCalls});
        if (reply.usage.totalTokens > 0) {
            emit(context, events,
                 Event{EventKind::usageChanged, {}, {}, reply.usage});
        }
        emit(context, events, Event{EventKind::messageDone});
        if (reply.toolCalls.empty()) {
            emit(context, events, Event{EventKind::finished});
            return;  // 没有工具请求时，模型回答就是本轮最终结果
        }

        bool progressed = !reply.content.empty();
        for (const model::ToolCall& call : reply.toolCalls) {
            bool stop = false;
            if (handleMeta(context, events, approvals, call, stop)) {
                progressed = true;
                if (stop) {
                    return;
                }
                continue;
            }

            std::string preview = tools_->preview(call);
            std::string approvalKey = call.function.name + '\0' + preview;
            bool approved = true;
            if (tools_->needsApproval(call) && !isApproved(approvalKey)) {
                countApproval();
                emit(context, events,
                     Event{EventKind::approval, preview, call});
                std::optional<ApprovalDecision> decision;
                try {
                    decision = approvals ? approvals(context)
                                         : std::optional<ApprovalDecision>{};
                } catch (const model::Cancelled&) {
                    emit(context, events,
                         Event{EventKind::finished, "Cancelled"});
                    return;
                }
                if (context.cancelled()) {
                    emit(context, events,
                         Event{EventKind::finished, "Cancelled"});
                    return;
                }
                approved = decision.has_value() && decision->approved;
                if (approved && decision->remember) {
                    rememberApproval(approvalKey);
                }
            }

            if (!approved) {
                refuseCall(context, events, call, "User denied this action.");
                continue;  // 拒绝本身不算环境进展，模型仍可改用别的工具
            }

            emit(context, events,
                 Event{EventKind::toolStarted, preview, call});
            model::Observation observation;
            bool failed = false;
            std::string output;
            try {
                observation = tools_->execute(context, call);
                output = observation.text;
            } catch (const model::Cancelled&) {
                emit(context, events,
                     Event{EventKind::finished, "Cancelled"});
                return;
            } catch (const std::exception& failure) {
                failed = true;
                output = std::string("Error: ") + failure.what();
            }
            if (context.cancelled()) {
                emit(context, events,
                     Event{EventKind::finished, "Cancelled"});
                return;  // 工具在后台返回后仍可能刚好收到取消信号
            }
            countToolRun(failed);
            progressed = true;
            if (failed && !observation.text.empty()) {
                output = observation.text + "\n" + output;
            }
            if (failed) {
                output = trim(output);
            }
            appendMessage(model::Message{"tool", output, observation.images,
                                         {}, call.id});
            if (recordToolOutcome(call, output, failed) &&
                requestRepeatedFailureClarification(context, events,
                                                    approvals, call, output)) {
                return;
            }
            emit(context, events,
                 Event{EventKind::toolDone, output, call, {}, failed});
        }

        if (progressed) {
            stalledTurns = 0;
        } else {
            ++stalledTurns;
            if (stalledTurns >= maxStalledTurns) {
                emit(context, events,
                     Event{EventKind::failed,
                           "Agent made no progress for several turns.", {},
                           {}, true});
                return;
            }
        }
    }

    emit(context, events,
         Event{EventKind::failed,
               "Agent stopped after 32 reasoning turns.", {}, {}, true});
}

std::string Agent::workspaceDiff(model::Context& context) {
    auto differ = std::dynamic_pointer_cast<WorkspaceDiffer>(tools_);
    if (!differ) {
        throw std::runtime_error("workspace diff is unavailable");
    }
    return differ->diff(context);
}

std::string Agent::undoLastChange() {
    auto undoer = std::dynamic_pointer_cast<WorkspaceUndoer>(tools_);
    if (!undoer) {
        throw std::runtime_error("workspace undo is unavailable");
    }
    return undoer->undo();
}

}  // namespace agentcore::agent
