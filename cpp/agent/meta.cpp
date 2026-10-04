/*
澄清元工具：把模型缺少的信息变成一次明确的用户询问，并把答案回填会话。
重复失败也通过同一条路径暂停，避免认知循环悄悄地无限重试同一个动作。
*/
#include "agent.hpp"

#include <json/json.h>

#include <cctype>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace agentcore::agent {
namespace {

constexpr const char* clarificationName = "request_clarification";

model::Tool clarificationTool() {
    model::Tool tool;
    tool.function.name = clarificationName;
    tool.function.description =
        "Pause and ask the user for missing information before continuing.";
    tool.function.parameters["type"] = "object";
    tool.function.parameters["required"] = Json::Value(Json::arrayValue);
    tool.function.parameters["required"].append("question");
    tool.function.parameters["properties"]["question"]["type"] = "string";
    tool.function.parameters["properties"]["options"]["type"] = "array";
    tool.function.parameters["properties"]["options"]["items"]["type"] =
        "string";
    return tool;
}

void emit(model::Context& context, const EventSink& events, Event event) {
    if (!events) return;
    if (event.kind != EventKind::finished &&
        event.kind != EventKind::failed && context.cancelled()) {
        return;
    }
    events(event);
}

std::optional<ApprovalDecision> receiveDecision(model::Context& context,
                                                const ApprovalSource& approvals) {
    if (!approvals) {
        return std::nullopt;  // 没有用户通道时，安全默认是拒绝
    }
    return approvals(context);
}

bool readClarification(const std::string& text,
                       std::string& question,
                       std::string& failure) {
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::Value value;
    std::string parseErrors;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(text.data(), text.data() + text.size(), &value,
                       &parseErrors)) {
        failure = "Clarification request failed: " + parseErrors;
        return false;
    }
    question = value.get("question", "").asString();
    if (question.empty()) {
        question = "Please provide the missing information.";
    }
    return true;
}

}  // namespace

std::vector<model::Tool> Agent::toolDefinitions() const {
    std::vector<model::Tool> definitions;
    definitions.push_back(clarificationTool());
    for (const model::Tool& tool : tools_->definitions()) {
        if (tool.function.name == clarificationName) {
            continue;  // 元工具由核心拥有，避免调用方覆盖它的语义
        }
        definitions.push_back(tool);
    }
    return definitions;
}

bool Agent::handleMeta(model::Context& context,
                       const EventSink& events,
                       const ApprovalSource& approvals,
                       const model::ToolCall& call,
                       bool& stop) {
    if (call.function.name != clarificationName) {
        return false;
    }
    std::string output;
    bool valid = requestClarification(context, events, approvals, call, output,
                                      stop);
    appendMessage(model::Message{"tool", output, {}, {}, call.id});
    recordToolOutcome(call, output, !valid);
    return true;
}

bool Agent::requestClarification(model::Context& context,
                                 const EventSink& events,
                                 const ApprovalSource& approvals,
                                 const model::ToolCall& call,
                                 std::string& output,
                                 bool& stop) {
    std::string question;
    std::string failure;
    if (!readClarification(call.function.arguments, question, failure)) {
        output = std::move(failure);
        return false;  // 坏参数回填给模型，让它自行修正请求
    }

    emit(context, events,
         Event{EventKind::clarification, question, call});
    std::optional<ApprovalDecision> decision;
    try {
        decision = receiveDecision(context, approvals);
    } catch (const model::Cancelled&) {
        emit(context, events, Event{EventKind::finished, "Cancelled"});
        stop = true;
        return false;
    }
    if (context.cancelled()) {
        emit(context, events, Event{EventKind::finished, "Cancelled"});
        stop = true;
        return false;
    }
    if (!decision || !decision->approved) {
        output = "User dismissed the clarification.";
        return true;
    }
    std::string answer = decision->answer;
    while (!answer.empty() && std::isspace(static_cast<unsigned char>(answer.back()))) {
        answer.pop_back();
    }
    std::size_t first = 0;
    while (first < answer.size() &&
           std::isspace(static_cast<unsigned char>(answer[first]))) {
        ++first;
    }
    answer = answer.substr(first);
    output = answer.empty() ? "User answered the clarification."
                            : "User answered: " + answer;
    return true;
}

bool Agent::requestRepeatedFailureClarification(
    model::Context& context,
    const EventSink& events,
    const ApprovalSource& approvals,
    const model::ToolCall& failedCall,
    const std::string&) {
    std::string question = "The same action failed repeatedly: " +
                           failedCall.function.name + ". How should I proceed?";
    model::ToolCall call;
    call.id = "clarify_repeated_failure";
    call.function.name = clarificationName;
    Json::Value arguments(Json::objectValue);
    arguments["question"] = question;
    arguments["options"] = Json::Value(Json::arrayValue);
    arguments["options"].append("Retry with a different approach");
    arguments["options"].append("Stop and explain the blocker");
    arguments["options"].append("Show the full output");
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    call.function.arguments = Json::writeString(writer, arguments);

    appendMessage(model::Message{"assistant", {}, {}, {call}});
    emit(context, events, Event{EventKind::clarification, question, call});
    std::optional<ApprovalDecision> decision;
    try {
        decision = receiveDecision(context, approvals);
    } catch (const model::Cancelled&) {
        emit(context, events, Event{EventKind::finished, "Cancelled"});
        return true;
    }
    if (context.cancelled()) {
        emit(context, events, Event{EventKind::finished, "Cancelled"});
        return true;
    }

    std::string answer = "User dismissed the clarification.";
    if (decision && decision->approved) {
        std::string supplied = decision->answer;
        std::size_t first = supplied.find_first_not_of(" \t\r\n");
        if (first != std::string::npos) {
            supplied = supplied.substr(first);
            supplied.erase(supplied.find_last_not_of(" \t\r\n") + 1);
            if (!supplied.empty()) answer = "User answered: " + supplied;
        }
    }
    appendMessage(model::Message{"tool", answer, {}, {}, call.id});
    recordToolOutcome(call, answer, false);  // 用户回答打破失败连续计数
    return false;
}

}  // namespace agentcore::agent
