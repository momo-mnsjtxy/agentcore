/*
会话状态：保存消息、审批记忆和运行统计，并在工具图片过多时收紧上下文。
这些动作集中在这里，让主循环只呈现用户目标如何向前推进。
*/
#include "agent.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>

namespace agentcore::agent {
namespace {

constexpr int keepRecentImages = 3;  // 模型只需要最近几张实际画面

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

std::size_t codePointSize(unsigned char firstByte) {
    if ((firstByte & 0x80U) == 0) {
        return 1;
    }
    if ((firstByte & 0xE0U) == 0xC0U) {
        return 2;
    }
    if ((firstByte & 0xF0U) == 0xE0U) {
        return 3;
    }
    if ((firstByte & 0xF8U) == 0xF0U) {
        return 4;
    }
    return 1;  // 损坏的 UTF-8 也要能产生稳定的失败摘要
}

std::string truncateCodePoints(std::string_view text, int limit) {
    if (limit <= 0) {
        return {};
    }
    std::size_t position = 0;
    int count = 0;
    while (position < text.size() && count < limit) {
        std::size_t width = codePointSize(
            static_cast<unsigned char>(text[position]));
        width = std::min(width, text.size() - position);
        position += width;
        ++count;
    }
    if (position == text.size()) {
        return std::string(text);  // 内容没有超过上限时保留完整错误首行
    }
    if (limit == 1) {
        return "…";  // 一个码点的预算只留下截断标记
    }
    position = 0;
    count = 0;
    while (position < text.size() && count < limit - 1) {
        std::size_t width = codePointSize(
            static_cast<unsigned char>(text[position]));
        width = std::min(width, text.size() - position);
        position += width;
        ++count;
    }
    std::string result(text.substr(0, position));
    result += "…";
    return result;
}

std::string firstTaskLine(std::string_view output) {
    std::size_t lineEnd = output.find('\n');
    std::string line = trim(output.substr(0, lineEnd));
    return truncateCodePoints(line, 120);
}

void pruneImages(std::vector<model::Message>& messages) {
    int seen = 0;
    for (auto message = messages.rbegin(); message != messages.rend(); ++message) {
        if (message->images.empty()) {
            continue;
        }
        ++seen;
        if (seen > keepRecentImages) {
            message->images.clear();  // 旧画面不应挤掉当前工具结果
        }
    }
}

void emit(model::Context& context, const EventSink& events, Event event) {
    if (!events) {
        return;
    }
    if (event.kind != EventKind::finished &&
        event.kind != EventKind::failed && context.cancelled()) {
        return;
    }
    events(event);
}

}  // namespace

void Agent::reset() {
    std::lock_guard lock(mutex_);
    history_.assign(1, system_);  // 新会话仍使用原来的项目指令
    approved_.clear();
    turns_ = 0;
    toolRuns_ = 0;
    approvals_ = 0;
    toolFailures_ = 0;
    failureKey_.clear();
    failureStreak_ = 0;
}

Stats Agent::stats() const {
    std::lock_guard lock(mutex_);
    return Stats{turns_, static_cast<int>(history_.size()) - 1, toolRuns_,
                 approvals_, toolFailures_};
}

Snapshot Agent::snapshot() const {
    std::lock_guard lock(mutex_);
    return Snapshot{std::vector<model::Message>(history_.begin() + 1,
                                                history_.end()),
                    turns_};
}

void Agent::restore(const Snapshot& snapshotValue) {
    std::lock_guard lock(mutex_);
    history_.clear();
    history_.push_back(system_);
    history_.insert(history_.end(), snapshotValue.messages.begin(),
                    snapshotValue.messages.end());
    turns_ = std::max(0, snapshotValue.turns);
    approved_.clear();  // 审批许可不能跨越一次恢复继续生效
    toolRuns_ = 0;
    approvals_ = 0;
    toolFailures_ = 0;
    failureKey_.clear();
    failureStreak_ = 0;
}

void Agent::appendMessage(const model::Message& message) {
    std::lock_guard lock(mutex_);
    history_.push_back(message);
    if (!message.images.empty()) {
        pruneImages(history_);
    }
}

std::vector<model::Message> Agent::startTurn() {
    std::lock_guard lock(mutex_);
    ++turns_;  // 每次真正发给模型的请求都计为一轮推理
    return history_;
}

void Agent::rememberApproval(const std::string& key) {
    std::lock_guard lock(mutex_);
    approved_[key] = true;
}

bool Agent::isApproved(const std::string& key) const {
    std::lock_guard lock(mutex_);
    return approved_.find(key) != approved_.end();
}

void Agent::countApproval() {
    std::lock_guard lock(mutex_);
    ++approvals_;
}

void Agent::countToolRun(bool failed) {
    std::lock_guard lock(mutex_);
    ++toolRuns_;
    if (failed) {
        ++toolFailures_;
    }
}

bool Agent::recordToolOutcome(const model::ToolCall& call,
                              const std::string& output,
                              bool failed) {
    std::lock_guard lock(mutex_);
    if (!failed) {
        failureKey_.clear();
        failureStreak_ = 0;
        return false;
    }
    std::string key = call.function.name + '\0' + firstTaskLine(output);
    if (key == failureKey_) {
        ++failureStreak_;
    } else {
        failureKey_ = std::move(key);
        failureStreak_ = 1;
    }
    return failureStreak_ >= 3;
}

void Agent::refuseCall(model::Context& context,
                       const EventSink& events,
                       const model::ToolCall& call,
                       std::string reason) {
    appendMessage(model::Message{"tool", reason, {}, {}, call.id});
    recordToolOutcome(call, reason, false);  // 拒绝不是工具失败
    emit(context, events,
         Event{EventKind::toolDone, std::move(reason), call, {}, false});
}

}  // namespace agentcore::agent
