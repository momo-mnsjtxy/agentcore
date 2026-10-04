/*
上下文压缩：把较旧的完整回合变成有界事实检查点，并保留最近回合原始结构。
切分只发生在 user 消息边界，避免把工具请求和工具结果拆成无法理解的半回合。
*/
#include "agent.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace agentcore::agent {
namespace {

constexpr int summaryLimit = 12'000;  // 检查点有界，避免压缩反而膨胀

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
    if ((firstByte & 0x80U) == 0) return 1;
    if ((firstByte & 0xE0U) == 0xC0U) return 2;
    if ((firstByte & 0xF0U) == 0xE0U) return 3;
    if ((firstByte & 0xF8U) == 0xF0U) return 4;
    return 1;
}

int codePointCount(std::string_view text) {
    int count = 0;
    for (std::size_t position = 0; position < text.size(); ++count) {
        std::size_t width = std::min(
            codePointSize(static_cast<unsigned char>(text[position])),
            text.size() - position);
        position += width;
    }
    return count;
}

std::string truncateCodePoints(std::string_view text, int limit) {
    if (limit <= 0) return {};
    std::size_t position = 0;
    int count = 0;
    while (position < text.size() && count < limit) {
        position += std::min(
            codePointSize(static_cast<unsigned char>(text[position])),
            text.size() - position);
        ++count;
    }
    if (position == text.size()) return std::string(text);
    if (limit == 1) return "…";
    position = 0;
    count = 0;
    while (position < text.size() && count < limit - 1) {
        position += std::min(
            codePointSize(static_cast<unsigned char>(text[position])),
            text.size() - position);
        ++count;
    }
    return std::string(text.substr(0, position)) + "…";
}

void appendSummary(std::string& summary,
                   std::string_view label,
                   std::string_view content,
                   int limit) {
    std::string clean = trim(content);
    if (clean.empty() || codePointCount(summary) >= summaryLimit) {
        return;
    }
    std::string block = "\n" + std::string(label) + ":\n" +
                        truncateCodePoints(clean, limit) + "\n";
    int remaining = summaryLimit - codePointCount(summary);
    summary += truncateCodePoints(block, remaining);
}

std::string summarize(const std::vector<model::Message>& messages) {
    std::unordered_map<std::string, std::string> toolNames;
    for (const model::Message& message : messages) {
        if (message.role != "assistant") continue;
        for (const model::ToolCall& call : message.toolCalls) {
            toolNames[call.id] = call.function.name;
        }
    }

    std::string summary =
        "Conversation checkpoint from earlier completed turns. Treat this as "
        "factual history; preserve the recent messages that follow it.\n";
    for (const model::Message& message : messages) {
        if (message.role == "system") {
            appendSummary(summary, "Earlier checkpoint", message.content, 2'500);
        } else if (message.role == "user") {
            appendSummary(summary, "User", message.content, 1'200);
        } else if (message.role == "assistant") {
            appendSummary(summary, "Assistant", message.content, 1'800);
            for (const model::ToolCall& call : message.toolCalls) {
                appendSummary(summary, "Tool request " + call.function.name,
                              call.function.arguments, 900);
            }
        } else if (message.role == "tool") {
            auto found = toolNames.find(message.toolCallID);
            std::string name = found == toolNames.end() ? "unknown" : found->second;
            appendSummary(summary, "Tool result " + name, message.content, 1'400);
        }
        if (codePointCount(summary) >= summaryLimit) break;
    }
    return truncateCodePoints(summary, summaryLimit);
}

std::size_t recentTurnStart(const std::vector<model::Message>& messages,
                            int keepUserTurns) {
    int seen = 0;
    for (std::size_t index = messages.size(); index-- > 1;) {
        if (messages[index].role != "user") continue;
        ++seen;
        if (seen == keepUserTurns) return index;
    }
    return 0;
}

}  // namespace

CompactResult Agent::compact(int keepUserTurns) {
    keepUserTurns = std::max(1, keepUserTurns);
    std::lock_guard lock(mutex_);
    int before = static_cast<int>(history_.size()) - 1;
    std::size_t cut = recentTurnStart(history_, keepUserTurns);
    if (cut <= 1) {
        return CompactResult{before, before, 0};
    }

    std::vector<model::Message> older(history_.begin() + 1,
                                      history_.begin() + cut);
    std::vector<model::Message> recent(history_.begin() + cut,
                                       history_.end());
    std::vector<model::Message> compacted;
    compacted.reserve(recent.size() + 2);
    compacted.push_back(system_);
    compacted.push_back(
        model::Message{"system", summarize(older)});  // 旧回合变成事实检查点
    compacted.insert(compacted.end(), recent.begin(), recent.end());
    history_ = std::move(compacted);

    int after = static_cast<int>(history_.size()) - 1;
    return CompactResult{before, after, before - after};
}

}  // namespace agentcore::agent
