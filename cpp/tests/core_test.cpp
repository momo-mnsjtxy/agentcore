/*
C++ 核心回归样例：用内存 Provider 和 Toolbox 验证认知循环，再用固定 SSE 向量验证协议。
测试只覆盖跨模块契约；细节仍由各业务文件直接表达，便于失败时定位。
*/
#include "../agent/agent.hpp"
#include "../openai/openai.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using agentcore::agent::Agent;
using agentcore::agent::ApprovalDecision;
using agentcore::agent::Event;
using agentcore::agent::EventKind;
using agentcore::agent::Toolbox;
using agentcore::model::Context;
using agentcore::model::Message;
using agentcore::model::Observation;
using agentcore::model::Provider;
using agentcore::model::Reply;
using agentcore::model::Stream;
using agentcore::model::Tool;
using agentcore::model::ToolCall;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class Answers final : public Provider {
public:
    std::vector<Reply> replies;
    std::vector<std::vector<Tool>> seenTools;
    int calls = 0;

    Reply complete(Context&,
                   const std::vector<Message>&,
                   const std::vector<Tool>& tools,
                   const Stream& stream) override {
        seenTools.push_back(tools);
        ++calls;
        if (calls <= static_cast<int>(replies.size())) {
            const Reply& reply = replies[calls - 1];
            if (stream && !reply.content.empty()) {
                stream({agentcore::model::StreamKind::outputDelta,
                        reply.content, {}});
            }
            return reply;
        }
        return Reply{"done", {}, {}, {}};
    }
};

class Files final : public Toolbox {
public:
    int executions = 0;

    std::vector<Tool> definitions() const override {
        Tool tool;
        tool.function.name = "write_file";
        tool.function.description = "Write a file";
        tool.function.parameters["type"] = "object";
        return {tool};
    }

    agentcore::agent::Capability capability(const ToolCall&) const override {
        return agentcore::agent::Capability::write;
    }

    bool needsApproval(const ToolCall&) const override { return true; }

    std::string preview(const ToolCall&) const override {
        return "write README.md";
    }

    Observation execute(Context&, const ToolCall&) override {
        ++executions;
        return Observation{"written", {}};
    }
};

ToolCall writeCall() {
    ToolCall call;
    call.id = "call_1";
    call.function.name = "write_file";
    call.function.arguments = R"({"path":"README.md"})";
    return call;
}

void testAgentLoop() {
    auto answers = std::make_shared<Answers>();
    answers->replies.push_back(Reply{"", {writeCall()}, {}, {}});
    auto files = std::make_shared<Files>();
    Agent brain(answers, files, "test rules");
    Context context;
    std::vector<Event> events;
    brain.run(context, "ship it", [&](const Event& event) { events.push_back(event); },
              [](Context&) {
                  return std::optional<ApprovalDecision>{ApprovalDecision{true, true, {}}};
              });

    require(files->executions == 1, "approved tool was not executed");
    require(brain.stats().tools == 1, "tool count was not recorded");
    require(!answers->seenTools.empty() && answers->seenTools[0].size() == 2,
            "clarification meta tool was not advertised");
    require(!events.empty() && events.back().kind == EventKind::finished,
            "agent did not finish after the answer");
}

void testStreams() {
    std::string chat =
        R"(data: {"choices":[{"delta":{"content":"I will inspect."}}]})"
        "\n\n"
        R"(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_","arguments":"{\"path\":"}}]}}]})"
        "\n\n"
        R"(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"\"README.md\"}"}}]}}]})"
        "\n\n"
        "data: [DONE]\n\n";
    auto parsed = agentcore::openai::readChatStream(chat);
    require(parsed.reply.content == "I will inspect.", "chat text mismatch");
    require(parsed.reply.toolCalls.size() == 1,
            "chat tool call count mismatch");
    require(parsed.reply.toolCalls[0].function.name == "read_file" &&
                parsed.reply.toolCalls[0].function.arguments ==
                    R"({"path":"README.md"})",
            "chat fragments were not joined");

    std::string responses =
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"Checking files\"}\n\n"
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"I will inspect.\"}\n\n"
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"read_file\",\"arguments\":\"\"}}\n\n"
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"path\\\":\\\"README.md\\\"}\"}\n\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":120,\"output_tokens\":30,\"total_tokens\":150}}}\n\n";
    auto complete = agentcore::openai::readResponsesStream(responses);
    require(complete.reply.responseID == "resp_1" &&
                complete.reply.usage.totalTokens == 150,
            "responses completion metadata mismatch");
    require(complete.reply.toolCalls.size() == 1 &&
                complete.reply.toolCalls[0].function.name == "read_file",
            "responses tool call mismatch");

    agentcore::model::Image image{"image/png", {'p', 'n', 'g'}};
    require(agentcore::openai::dataURL(image) == "data:image/png;base64,cG5n",
            "image data URL mismatch");
}

void testCompaction() {
    auto answers = std::make_shared<Answers>();
    auto files = std::make_shared<Files>();
    Agent brain(answers, files, "rules");
    agentcore::agent::Snapshot snapshot;
    for (int turn = 1; turn <= 4; ++turn) {
        snapshot.messages.push_back(Message{"user", "goal " + std::to_string(turn)});
        snapshot.messages.push_back(Message{"assistant", "answer"});
    }
    snapshot.turns = 4;
    brain.restore(snapshot);
    auto result = brain.compact(1);
    require(result.removedMessages > 0, "old turns were not compacted");
    require(brain.snapshot().messages[0].role == "system",
            "compaction lost its factual checkpoint");
}

}  // namespace

int main() {
    try {
        testAgentLoop();
        testStreams();
        testCompaction();
        std::cout << "agentcore C++ tests passed\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
