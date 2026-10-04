/*
Chat Completions 协议：把会话和工具编成请求，并把增量文本与工具调用拼回 Reply。
Responses 的字段映射在 responses.cpp；Client::complete 只负责按协议路由。
*/
#include "openai.hpp"
#include "detail.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace agentcore::openai {
namespace {

Json::Value imageContent(const model::Message& message) {
    Json::Value parts(Json::arrayValue);
    if (!message.content.empty()) {
        Json::Value text(Json::objectValue);
        text["type"] = "text";
        text["text"] = message.content;
        parts.append(text);
    }
    for (const model::Image& image : message.images) {
        Json::Value picture(Json::objectValue);
        picture["type"] = "image_url";
        picture["image_url"]["url"] = dataURL(image);
        parts.append(picture);
    }
    return parts;
}

Json::Value chatMessages(const std::vector<model::Message>& messages) {
    Json::Value result(Json::arrayValue);
    for (const model::Message& message : messages) {
        Json::Value item(Json::objectValue);
        item["role"] = message.role;
        if (!message.toolCallID.empty()) {
            item["tool_call_id"] = message.toolCallID;
        }
        if (!message.toolCalls.empty()) {
            item["tool_calls"] = Json::Value(Json::arrayValue);
            for (const model::ToolCall& call : message.toolCalls) {
                item["tool_calls"].append(model::toJson(call));
            }
        }
        if (message.images.empty()) {
            item["content"] = message.content;
            result.append(item);
            continue;
        }
        if (message.role == "tool") {
            item["content"] = message.content;
            result.append(item);  // 工具文字仍保留原 tool 角色
            Json::Value pictureMessage(Json::objectValue);
            pictureMessage["role"] = "user";
            pictureMessage["content"] = imageContent(message);
            result.append(pictureMessage);  // 图片用后续 user part 兼容旧服务
            continue;
        }
        item["content"] = imageContent(message);
        result.append(item);
    }
    return result;
}

Json::Value chatTools(const std::vector<model::Tool>& tools) {
    Json::Value result(Json::arrayValue);
    for (const model::Tool& tool : tools) {
        result.append(model::toJson(tool));
    }
    return result;
}

void stream(const model::Stream& callback, const model::StreamEvent& event) {
    if (callback) callback(event);
}

}  // namespace

Json::Value chatPayload(const std::string& modelName,
                        const std::vector<model::Message>& messages,
                        const std::vector<model::Tool>& tools) {
    Json::Value payload(Json::objectValue);
    payload["model"] = modelName;
    payload["messages"] = chatMessages(messages);
    payload["tools"] = chatTools(tools);
    payload["tool_choice"] = "auto";
    payload["stream_options"]["include_usage"] = true;
    payload["stream"] = true;
    return payload;
}

ParsedReply readChatStream(std::string_view source,
                           const model::Stream& callback) {
    ParsedReply parsed;
    detail::eachDataLine(source, [&](std::string_view data) {
        if (data == "[DONE]") return;
        Json::Value chunk = detail::parseJson(data, "model stream");
        if (chunk.isMember("error")) {
            throw std::runtime_error("model stream: " +
                                     chunk["error"].get("message", "").asString());
        }
        const Json::Value& usage = chunk["usage"];
        if (usage.isObject() && usage.get("total_tokens", 0).asInt() > 0) {
            parsed.progressed = true;
            parsed.reply.usage.inputTokens = usage.get("prompt_tokens", 0).asInt();
            parsed.reply.usage.outputTokens = usage.get("completion_tokens", 0).asInt();
            parsed.reply.usage.totalTokens = usage.get("total_tokens", 0).asInt();
            stream(callback, model::StreamEvent{model::StreamKind::usageChanged,
                                                {}, parsed.reply.usage});
        }
        const Json::Value& choices = chunk["choices"];
        if (!choices.isArray() || choices.empty()) return;
        const Json::Value& delta = choices[0]["delta"];
        std::string content = delta.get("content", "").asString();
        if (!content.empty()) {
            parsed.progressed = true;
            parsed.reply.content += content;
            stream(callback,
                   model::StreamEvent{model::StreamKind::outputDelta, content});
        }
        for (const Json::Value& incoming : delta["tool_calls"]) {
            parsed.progressed = true;
            int index = incoming.get("index", 0).asInt();
            if (index < 0) continue;
            while (static_cast<int>(parsed.reply.toolCalls.size()) <= index) {
                parsed.reply.toolCalls.push_back(model::ToolCall{});
            }
            model::ToolCall& call = parsed.reply.toolCalls[index];
            std::string id = incoming.get("id", "").asString();
            std::string type = incoming.get("type", "").asString();
            if (!id.empty()) call.id = id;
            if (!type.empty()) call.type = type;
            const Json::Value& function = incoming["function"];
            call.function.name += function.get("name", "").asString();
            call.function.arguments += function.get("arguments", "").asString();
        }
    });
    return parsed;
}

Client::Client(std::string baseURL,
               std::string apiKey,
               std::string modelName,
               std::string protocol,
               std::string reasoning)
    : baseURL_(std::move(baseURL)),
      apiKey_(std::move(apiKey)),
      modelName_(std::move(modelName)),
      protocol_(std::move(protocol)),
      reasoning_(std::move(reasoning)) {}

model::Reply Client::complete(model::Context& context,
                              const std::vector<model::Message>& messages,
                              const std::vector<model::Tool>& tools,
                              const model::Stream& streamCallback) {
    if (protocol_ == "responses") {
        Json::Value payload =
            responsesPayload(modelName_, reasoning_, messages, tools);
        return completeStream(
            context, "/responses", payload, streamCallback,
            [](std::string_view body, const model::Stream& stream) {
                return readResponsesStream(body, stream);
            });
    }
    Json::Value payload = chatPayload(modelName_, messages, tools);
    return completeStream(
        context, "/chat/completions", payload, streamCallback,
        [](std::string_view body, const model::Stream& stream) {
            return readChatStream(body, stream);
        });
}

}  // namespace agentcore::openai
