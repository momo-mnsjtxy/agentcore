/*
Responses 协议：把会话映射为 input items，并解析文本、推理、工具和用量事件。
每轮显式发送当前会话并保持 store=false，因而兼容 OpenAI-compatible 本地代理。
*/
#include "openai.hpp"
#include "detail.hpp"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace agentcore::openai {
namespace {

Json::Value responseContent(std::string_view textType,
                            const model::Message& message) {
    Json::Value parts(Json::arrayValue);
    if (!message.content.empty()) {
        Json::Value text(Json::objectValue);
        text["type"] = std::string(textType);
        text["text"] = message.content;
        parts.append(text);
    }
    for (const model::Image& image : message.images) {
        Json::Value picture(Json::objectValue);
        picture["type"] = "input_image";
        picture["image_url"] = dataURL(image);
        parts.append(picture);
    }
    if (parts.empty()) {
        Json::Value text(Json::objectValue);
        text["type"] = std::string(textType);
        text["text"] = "";
        parts.append(text);
    }
    return parts;
}

Json::Value responseOutput(const model::Message& message) {
    if (message.images.empty()) {
        return Json::Value(message.content);
    }
    Json::Value parts(Json::arrayValue);
    Json::Value text(Json::objectValue);
    text["type"] = "input_text";
    text["text"] = message.content;
    parts.append(text);
    for (const model::Image& image : message.images) {
        Json::Value picture(Json::objectValue);
        picture["type"] = "input_image";
        picture["image_url"] = dataURL(image);
        parts.append(picture);
    }
    return parts;
}

Json::Value responseInput(const std::vector<model::Message>& messages) {
    Json::Value input(Json::arrayValue);
    for (const model::Message& message : messages) {
        if (message.role == "user") {
            Json::Value item(Json::objectValue);
            item["type"] = "message";
            item["role"] = "user";
            item["content"] = responseContent("input_text", message);
            input.append(item);
        } else if (message.role == "assistant") {
            if (!message.content.empty() || !message.images.empty()) {
                Json::Value item(Json::objectValue);
                item["type"] = "message";
                item["role"] = "assistant";
                item["content"] = responseContent("output_text", message);
                input.append(item);
            }
            for (const model::ToolCall& call : message.toolCalls) {
                Json::Value item(Json::objectValue);
                item["type"] = "function_call";
                item["call_id"] = call.id;
                item["name"] = call.function.name;
                item["arguments"] = call.function.arguments;
                input.append(item);
            }
        } else if (message.role == "tool") {
            Json::Value item(Json::objectValue);
            item["type"] = "function_call_output";
            item["call_id"] = message.toolCallID;
            item["output"] = responseOutput(message);
            input.append(item);
        }
    }
    return input;
}

std::string instructions(const std::vector<model::Message>& messages) {
    std::string result;
    for (const model::Message& message : messages) {
        if (message.role != "system" || message.content.empty()) continue;
        if (!result.empty()) result += "\n\n";
        result += message.content;
    }
    return result;
}

Json::Value responseTools(const std::vector<model::Tool>& tools) {
    Json::Value result(Json::arrayValue);
    for (const model::Tool& tool : tools) {
        Json::Value value(Json::objectValue);
        value["type"] = "function";
        value["name"] = tool.function.name;
        value["description"] = tool.function.description;
        value["parameters"] = tool.function.parameters;
        value["strict"] = false;
        result.append(value);
    }
    return result;
}

}  // namespace

Json::Value responsesPayload(const std::string& modelName,
                             const std::string& reasoning,
                             const std::vector<model::Message>& messages,
                             const std::vector<model::Tool>& tools) {
    Json::Value payload(Json::objectValue);
    payload["model"] = modelName;
    payload["instructions"] = instructions(messages);
    payload["input"] = responseInput(messages);
    payload["tools"] = responseTools(tools);
    payload["tool_choice"] = "auto";
    payload["parallel_tool_calls"] = false;
    payload["store"] = false;
    payload["stream"] = true;
    if (!reasoning.empty() && reasoning != "none") {
        payload["reasoning"]["effort"] = reasoning;
        payload["reasoning"]["summary"] = "auto";
    }
    return payload;
}

namespace {

struct ResponseItem {
    std::string type;
    std::string id;
    std::string callID;
    std::string name;
    std::string arguments;
};

ResponseItem readItem(const Json::Value& value) {
    return ResponseItem{value.get("type", "").asString(),
                        value.get("id", "").asString(),
                        value.get("call_id", "").asString(),
                        value.get("name", "").asString(),
                        value.get("arguments", "").asString()};
}

std::string callKey(const ResponseItem& item) {
    return item.id.empty() ? item.callID : item.id;
}

void mergeCall(std::unordered_map<std::string, model::ToolCall>& calls,
               std::vector<std::string>& order,
               const ResponseItem& item) {
    std::string key = callKey(item);
    if (key.empty()) return;
    auto found = calls.find(key);
    if (found == calls.end()) {
        order.push_back(key);
        model::ToolCall call;
        call.type = "function";
        found = calls.emplace(key, std::move(call)).first;
    }
    model::ToolCall& call = found->second;
    if (!item.callID.empty()) call.id = item.callID;
    if (!item.name.empty()) call.function.name = item.name;
    if (!item.arguments.empty()) call.function.arguments = item.arguments;
}

void stream(const model::Stream& callback, const model::StreamEvent& event) {
    if (callback) callback(event);
}

std::string apiError(const Json::Value& value, std::string_view fallback) {
    if (!value.isObject()) return std::string(fallback);
    std::string message = value.get("message", "").asString();
    std::string code = value.get("code", "").asString();
    std::string result = std::string(fallback) + ": " + message;
    if (!code.empty()) result += " (" + code + ")";
    return result;
}

}  // namespace

ParsedReply readResponsesStream(std::string_view source,
                                const model::Stream& callback) {
    ParsedReply parsed;
    std::unordered_map<std::string, model::ToolCall> calls;
    std::vector<std::string> order;

    detail::eachDataLine(source, [&](std::string_view data) {
        if (data == "[DONE]") return;
        Json::Value event = detail::parseJson(data, "responses stream");
        std::string type = event.get("type", "").asString();
        const Json::Value& response = event["response"];
        if (type == "response.output_text.delta") {
            parsed.progressed = true;
            std::string text = event.get("delta", "").asString();
            parsed.reply.content += text;
            stream(callback,
                   model::StreamEvent{model::StreamKind::outputDelta, text});
        } else if (type == "response.reasoning_summary_text.delta" ||
                   type == "response.reasoning_text.delta") {
            parsed.progressed = true;
            std::string text = event.get("delta", "").asString();
            stream(callback, model::StreamEvent{model::StreamKind::reasoningDelta,
                                                text});
        } else if (type == "response.output_item.added" ||
                   type == "response.output_item.done") {
            ResponseItem item = readItem(event["item"]);
            if (item.type == "function_call") {
                parsed.progressed = true;
                mergeCall(calls, order, item);
            }
        } else if (type == "response.function_call_arguments.delta") {
            parsed.progressed = true;
            std::string itemID = event.get("item_id", "").asString();
            if (itemID.empty()) return;
            auto found = calls.find(itemID);
            if (found == calls.end()) {
                order.push_back(itemID);
                model::ToolCall call;
                call.type = "function";
                found = calls.emplace(itemID, std::move(call)).first;
            }
            found->second.function.arguments +=
                event.get("delta", "").asString();
        } else if (type == "response.completed") {
            parsed.progressed = true;
            for (const Json::Value& itemValue : response["output"]) {
                ResponseItem item = readItem(itemValue);
                if (item.type == "function_call") {
                    mergeCall(calls, order, item);
                }
            }
            parsed.reply.responseID = response.get("id", "").asString();
            const Json::Value& usage = response["usage"];
            parsed.reply.usage.inputTokens = usage.get("input_tokens", 0).asInt();
            parsed.reply.usage.outputTokens = usage.get("output_tokens", 0).asInt();
            parsed.reply.usage.totalTokens = usage.get("total_tokens", 0).asInt();
            if (parsed.reply.usage.totalTokens > 0) {
                stream(callback, model::StreamEvent{model::StreamKind::usageChanged,
                                                    {}, parsed.reply.usage});
            }
        } else if (type == "response.failed") {
            throw std::runtime_error(apiError(response["error"],
                                               "response failed"));
        } else if (type == "response.incomplete") {
            std::string reason = response["incomplete_details"]
                                     .get("reason", "unknown")
                                     .asString();
            throw std::runtime_error("incomplete response: " + reason);
        } else if (type == "error") {
            throw std::runtime_error(apiError(event["error"],
                                               "responses stream failed"));
        }
    });

    for (const std::string& key : order) {
        const model::ToolCall& call = calls[key];
        if (!call.id.empty() && !call.function.name.empty()) {
            parsed.reply.toolCalls.push_back(call);
        }
    }
    return parsed;
}

}  // namespace agentcore::openai
