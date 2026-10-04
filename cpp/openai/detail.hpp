/*
OpenAI 协议的共享小工具：负责 JSON 解码、SSE 行切分和图片编码。
它不承载业务决策；重试和两种协议的字段映射留在各自的实现文件。
*/
#pragma once

#include <functional>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <json/json.h>

namespace agentcore::openai::detail {

// trim 去掉协议行两侧空白，保留正文中的空格。
std::string trim(std::string_view text);

// parseJson 把一条 SSE 数据解码成对象，并把上下文加入错误信息。
Json::Value parseJson(std::string_view text, std::string_view context);

// compactJson 用无缩进形式生成请求体和元工具参数。
std::string compactJson(const Json::Value& value);

// eachDataLine 只把 data: 行交给协议解析器，忽略 SSE 注释和空行。
void eachDataLine(std::string_view source,
                  const std::function<void(std::string_view)>& consume);

// encodeBase64 用标准字母表编码图片字节。
std::string encodeBase64(const std::vector<std::uint8_t>& data);

}  // namespace agentcore::openai::detail
