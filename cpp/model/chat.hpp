/*
模型对话协议：定义 Agent、模型服务和工具之间交换的简单数据。
业务循环只依赖这些结构，因此更换模型供应商不会改变工具和会话代码。
调用示例：provider.complete(context, messages, tools, stream)。
*/
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <json/json.h>

namespace agentcore::model {

// Image 是送给模型查看的图片；会话快照只复制元数据，不强制写入磁盘。
struct Image {
    std::string mimeType;
    std::vector<std::uint8_t> data;
};

// FunctionCall 保存模型要求本地执行的函数和 JSON 参数。
struct FunctionCall {
    std::string name;
    std::string arguments;
};

// ToolCall 是模型在一轮回答中发起的一次工具调用。
struct ToolCall {
    std::string id;
    std::string type = "function";
    FunctionCall function;
};

// FunctionTool 描述一个函数工具的参数 JSON Schema。
struct FunctionTool {
    std::string name;
    std::string description;
    Json::Value parameters = Json::Value(Json::objectValue);
};

// Tool 是模型可见的本地能力描述。
struct Tool {
    std::string type = "function";
    FunctionTool function;
};

// Message 是一条模型能够理解的会话消息。
struct Message {
    std::string role;
    std::string content;
    std::vector<Image> images;
    std::vector<ToolCall> toolCalls;
    std::string toolCallID;

    Message() = default;
    Message(std::string roleValue, std::string contentValue)
        : role(std::move(roleValue)), content(std::move(contentValue)) {}
    Message(std::string roleValue,
            std::string contentValue,
            std::vector<Image> imageValues,
            std::vector<ToolCall> callValues,
            std::string toolCallIDValue = {})
        : role(std::move(roleValue)),
          content(std::move(contentValue)),
          images(std::move(imageValues)),
          toolCalls(std::move(callValues)),
          toolCallID(std::move(toolCallIDValue)) {}
};

// Observation 是工具完成后回填给模型的文字和图片。
struct Observation {
    std::string text;
    std::vector<Image> images;
};

// Usage 是最近一次模型回答报告的 token 用量。
struct Usage {
    int inputTokens = 0;
    int outputTokens = 0;
    int totalTokens = 0;
};

// Reply 是一轮流式推理收集完毕后的完整结果。
struct Reply {
    std::string content;
    std::vector<ToolCall> toolCalls;
    Usage usage;
    std::string responseID;
};

// StreamKind 区分文字、推理摘要和用量变化。
enum class StreamKind {
    outputDelta,
    reasoningDelta,
    usageChanged,
};

// StreamEvent 是模型向 Agent 发出的一次增量变化。
struct StreamEvent {
    StreamKind kind;
    std::string text;
    Usage usage;

    StreamEvent(StreamKind kindValue, std::string textValue)
        : kind(kindValue), text(std::move(textValue)) {}
    StreamEvent(StreamKind kindValue, std::string textValue, Usage usageValue)
        : kind(kindValue), text(std::move(textValue)), usage(usageValue) {}
};

// Cancelled 让调用方把用户取消与普通模型失败区分开。
class Cancelled final : public std::runtime_error {
public:
    Cancelled() : std::runtime_error("operation cancelled") {}
};

// Context 是一次运行的显式取消信号，工具和网络请求共享它。
class Context {
public:
    void cancel() noexcept { cancelled_.store(true, std::memory_order_relaxed); }

    bool cancelled() const noexcept {
        return cancelled_.load(std::memory_order_relaxed);
    }

    void throwIfCancelled() const {
        if (cancelled()) {
            throw Cancelled();
        }
    }

private:
    std::atomic_bool cancelled_{false};
};

// Stream 接收模型产生的增量，空回调表示调用方不需要显示过程。
using Stream = std::function<void(const StreamEvent&)>;

// Provider 完成一轮推理；实现可以是 OpenAI，也可以是本地或测试模型。
class Provider {
public:
    virtual ~Provider() = default;

    virtual Reply complete(Context& context,
                           const std::vector<Message>& messages,
                           const std::vector<Tool>& tools,
                           const Stream& stream) = 0;
};

// toJson 只负责公共协议数据的可预测编码，供 HTTP 层和检查点使用。
inline Json::Value toJson(const FunctionCall& call) {
    Json::Value value(Json::objectValue);
    value["name"] = call.name;
    value["arguments"] = call.arguments;
    return value;
}

inline Json::Value toJson(const ToolCall& call) {
    Json::Value value(Json::objectValue);
    value["id"] = call.id;
    value["type"] = call.type;
    value["function"] = toJson(call.function);
    return value;
}

inline Json::Value toJson(const Tool& tool) {
    Json::Value value(Json::objectValue);
    value["type"] = tool.type;
    value["function"]["name"] = tool.function.name;
    value["function"]["description"] = tool.function.description;
    value["function"]["parameters"] = tool.function.parameters;
    return value;
}

inline Json::Value toJson(const Usage& usage) {
    Json::Value value(Json::objectValue);
    value["input_tokens"] = usage.inputTokens;
    value["output_tokens"] = usage.outputTokens;
    value["total_tokens"] = usage.totalTokens;
    return value;
}

}  // namespace agentcore::model
