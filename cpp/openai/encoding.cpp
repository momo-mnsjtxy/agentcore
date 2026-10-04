/*
协议编码基础：把 SSE 文本切成数据行，把 JSON 安全地编解码，并准备图片 data URL。
这些操作没有模型业务判断，因此两条 OpenAI 协议共享同一份实现。
*/
#include "openai.hpp"
#include "detail.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace agentcore::openai::detail {

std::string trim(std::string_view text) {
    std::size_t first = 0;
    while (first < text.size() &&
           (text[first] == ' ' || text[first] == '\t' ||
            text[first] == '\r' || text[first] == '\n')) {
        ++first;
    }
    std::size_t last = text.size();
    while (last > first &&
           (text[last - 1] == ' ' || text[last - 1] == '\t' ||
            text[last - 1] == '\r' || text[last - 1] == '\n')) {
        --last;
    }
    return std::string(text.substr(first, last - first));
}

Json::Value parseJson(std::string_view text, std::string_view context) {
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::Value value;
    std::string errors;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(text.data(), text.data() + text.size(), &value,
                       &errors)) {
        throw std::runtime_error("decode " + std::string(context) + ": " +
                                 errors);
    }
    return value;
}

std::string compactJson(const Json::Value& value) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    return Json::writeString(builder, value);
}

void eachDataLine(std::string_view source,
                  const std::function<void(std::string_view)>& consume) {
    constexpr std::size_t maximumLine = 4 * 1024 * 1024;
    std::size_t position = 0;
    while (position <= source.size()) {
        std::size_t lineEnd = source.find('\n', position);
        if (lineEnd == std::string_view::npos) {
            lineEnd = source.size();
        }
        std::string_view line = source.substr(position, lineEnd - position);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        if (line.size() > maximumLine) {
            throw std::runtime_error("model stream line exceeds 4 MiB");
        }
        if (line.starts_with("data:")) {
            consume(trim(line.substr(5)));
        }
        if (lineEnd == source.size()) {
            break;
        }
        position = lineEnd + 1;
    }
}

std::string encodeBase64(const std::vector<std::uint8_t>& data) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((data.size() + 2) / 3 * 4);
    for (std::size_t index = 0; index < data.size(); index += 3) {
        std::uint32_t value = static_cast<std::uint32_t>(data[index]) << 16;
        if (index + 1 < data.size()) {
            value |= static_cast<std::uint32_t>(data[index + 1]) << 8;
        }
        if (index + 2 < data.size()) {
            value |= data[index + 2];
        }
        encoded.push_back(alphabet[(value >> 18) & 63]);
        encoded.push_back(alphabet[(value >> 12) & 63]);
        encoded.push_back(index + 1 < data.size() ? alphabet[(value >> 6) & 63]
                                                   : '=');
        encoded.push_back(index + 2 < data.size() ? alphabet[value & 63] : '=');
    }
    return encoded;
}

}  // namespace agentcore::openai::detail

namespace agentcore::openai {

std::string dataURL(const model::Image& image) {
    std::string mime = image.mimeType.empty() ? "image/png" : image.mimeType;
    return "data:" + mime + ";base64," + detail::encodeBase64(image.data);
}

}  // namespace agentcore::openai
