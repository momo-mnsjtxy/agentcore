/*
OpenAI 兼容连接：在 Responses 和 Chat Completions 之间路由流式推理。
Client 实现 model::Provider；调用方只需提供 base URL、密钥、模型和协议名。
*/
#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <json/json.h>

#include "../model/chat.hpp"

namespace agentcore::openai {

// ParsedReply 同时保留完整回答和“是否已有有效输出”这个重试判断。
struct ParsedReply {
    model::Reply reply;
    bool progressed = false;
};

// readChatStream 解析 Chat Completions 的 SSE 文本并转发增量事件。
ParsedReply readChatStream(std::string_view source,
                           const model::Stream& stream = {});

// readResponsesStream 解析 Responses 的 SSE 文本并转发增量事件。
ParsedReply readResponsesStream(std::string_view source,
                                const model::Stream& stream = {});

// chatPayload 和 responsesPayload 暴露可检查的请求形状，便于本地代理测试。
Json::Value chatPayload(const std::string& modelName,
                        const std::vector<model::Message>& messages,
                        const std::vector<model::Tool>& tools);
Json::Value responsesPayload(const std::string& modelName,
                             const std::string& reasoning,
                             const std::vector<model::Message>& messages,
                             const std::vector<model::Tool>& tools);

// dataURL 把工具截图编码成模型协议要求的内联图片。
std::string dataURL(const model::Image& image);

// Client 是一个可复用的 OpenAI-compatible 流式模型连接。
class Client final : public model::Provider {
public:
    Client(std::string baseURL,
           std::string apiKey,
           std::string modelName,
           std::string protocol = "responses",
           std::string reasoning = "none");

    // complete 完成一轮推理；只有 protocol 恰好为 responses 时走 Responses。
    model::Reply complete(model::Context& context,
                          const std::vector<model::Message>& messages,
                          const std::vector<model::Tool>& tools,
                          const model::Stream& stream) override;

private:
    using Decoder = std::function<ParsedReply(std::string_view,
                                               const model::Stream&)>;

    model::Reply completeStream(model::Context& context,
                                std::string_view path,
                                const Json::Value& payload,
                                const model::Stream& stream,
                                const Decoder& decoder);

    std::string baseURL_;
    std::string apiKey_;
    std::string modelName_;
    std::string protocol_;
    std::string reasoning_;
};

}  // namespace agentcore::openai
