/*
HTTP 传输：发送 JSON、收集 SSE、处理取消和有限重试。
只有在尚未看到任何有效模型输出时才重放请求，避免重复执行用户已经看到的回答。
*/
#include "openai.hpp"
#include "detail.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cctype>
#include <ctime>
#include <mutex>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace agentcore::openai {
namespace {

constexpr int maxAttempts = 3;  // 短暂网络故障最多再试两次

struct Transfer {
    std::string body;
    std::string retryAfter;
    model::Context* context = nullptr;
};

std::size_t writeBody(char* data, std::size_t size, std::size_t count,
                      void* target) {
    auto* transfer = static_cast<Transfer*>(target);
    std::size_t bytes = size * count;
    transfer->body.append(data, bytes);
    return bytes;
}

bool equalHeader(std::string_view actual, std::string_view expected) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::tolower(static_cast<unsigned char>(actual[index])) !=
            std::tolower(static_cast<unsigned char>(expected[index]))) {
            return false;
        }
    }
    return true;
}

std::size_t readHeader(char* data, std::size_t size, std::size_t count,
                       void* target) {
    auto* transfer = static_cast<Transfer*>(target);
    std::string_view line(data, size * count);
    std::size_t colon = line.find(':');
    if (colon != std::string_view::npos &&
        equalHeader(detail::trim(line.substr(0, colon)), "Retry-After")) {
        transfer->retryAfter = detail::trim(line.substr(colon + 1));
    }
    return size * count;
}

int checkProgress(void* target,
                  curl_off_t,
                  curl_off_t,
                  curl_off_t,
                  curl_off_t) {
    auto* context = static_cast<model::Context*>(target);
    return context && context->cancelled() ? 1 : 0;
}

struct HttpResult {
    CURLcode code = CURLE_OK;
    long status = 0;
    std::string body;
    std::string retryAfter;
    std::string error;
};

HttpResult perform(model::Context& context,
                   const std::string& url,
                   const std::string& body,
                   const std::string& apiKey) {
    static std::once_flag initialization;
    std::call_once(initialization, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });

    CURL* handle = curl_easy_init();
    if (!handle) throw std::runtime_error("unable to create HTTP client");

    Transfer transfer;
    transfer.context = &context;
    curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: text/event-stream");
    if (!apiKey.empty()) {
        std::string authorization = "Authorization: Bearer " + apiKey;
        headers = curl_slist_append(headers, authorization.c_str());
    }

    curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle, CURLOPT_POST, 1L);
    curl_easy_setopt(handle, CURLOPT_POSTFIELDS, body.data());
    curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(body.size()));
    curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, writeBody);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &transfer);
    curl_easy_setopt(handle, CURLOPT_HEADERFUNCTION, readHeader);
    curl_easy_setopt(handle, CURLOPT_HEADERDATA, &transfer);
    curl_easy_setopt(handle, CURLOPT_XFERINFOFUNCTION, checkProgress);
    curl_easy_setopt(handle, CURLOPT_XFERINFODATA, &context);
    curl_easy_setopt(handle, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, 10'000L);
    curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, 120'000L);
    curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(handle, CURLOPT_ACCEPT_ENCODING, "");

    HttpResult result;
    result.code = curl_easy_perform(handle);
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &result.status);
    result.body = std::move(transfer.body);
    result.retryAfter = std::move(transfer.retryAfter);
    if (result.code != CURLE_OK) {
        result.error = curl_easy_strerror(result.code);
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(handle);
    return result;
}

bool retryableStatus(long status) {
    return status == 429 || status == 500 || status == 502 || status == 503 ||
           status == 504;
}

bool retryableConnection(CURLcode code) {
    return code != CURLE_OK && code != CURLE_ABORTED_BY_CALLBACK &&
           code != CURLE_URL_MALFORMAT && code != CURLE_UNSUPPORTED_PROTOCOL &&
           code != CURLE_BAD_FUNCTION_ARGUMENT;
}

std::pair<std::chrono::milliseconds, bool> parseRetryAfter(
    std::string_view value) {
    std::string clean = detail::trim(value);
    if (clean.empty()) return {{}, false};
    char* end = nullptr;
    long seconds = std::strtol(clean.c_str(), &end, 10);
    if (end && *end == '\0' && seconds >= 0) {
        return {std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::seconds(seconds)),
                true};
    }
    time_t date = curl_getdate(clean.c_str(), nullptr);
    if (date < 0) return {{}, false};
    auto now = std::chrono::system_clock::now();
    auto target = std::chrono::system_clock::from_time_t(date);
    if (target <= now) return {std::chrono::milliseconds(0), true};
    return {std::chrono::duration_cast<std::chrono::milliseconds>(target - now),
            true};
}

std::chrono::milliseconds jitter(std::chrono::milliseconds limit) {
    if (limit.count() <= 0) return std::chrono::milliseconds(0);
    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<long long> distribution(0, limit.count());
    return std::chrono::milliseconds(distribution(generator));
}

void waitForRetry(model::Context& context,
                  int attempt,
                  std::chrono::milliseconds explicitDelay,
                  bool hasExplicitDelay) {
    auto delay = explicitDelay;
    if (!hasExplicitDelay) {
        auto base = std::chrono::milliseconds(250LL << attempt);
        delay = base + jitter(base / 2);
    }
    constexpr auto quantum = std::chrono::milliseconds(10);
    while (delay.count() > 0) {
        context.throwIfCancelled();
        auto pause = std::min(delay, quantum);
        std::this_thread::sleep_for(pause);
        delay -= pause;
    }
}

std::string retryMessage(int attempts, std::string message) {
    if (attempts <= 1) return message;
    return "model request failed after " + std::to_string(attempts) +
           " attempts: " + message;
}

std::string endpoint(std::string baseURL, std::string_view path) {
    while (!baseURL.empty() && baseURL.back() == '/') baseURL.pop_back();
    return baseURL + std::string(path);
}

bool bodyShowsProgress(std::string_view body) {
    return body.find("output_text.delta") != std::string_view::npos ||
           body.find("reasoning_summary_text.delta") != std::string_view::npos ||
           body.find("tool_calls") != std::string_view::npos ||
           body.find("function_call") != std::string_view::npos ||
           body.find("\"content\"") != std::string_view::npos;
}

}  // namespace

model::Reply Client::completeStream(model::Context& context,
                                    std::string_view path,
                                    const Json::Value& payload,
                                    const model::Stream& streamCallback,
                                    const Decoder& decoder) {
    std::string body = detail::compactJson(payload);
    std::string url = endpoint(baseURL_, path);
    std::string lastError;

    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
        context.throwIfCancelled();
        HttpResult response = perform(context, url, body, apiKey_);
        if (response.code != CURLE_OK) {
            if (response.code == CURLE_ABORTED_BY_CALLBACK ||
                context.cancelled()) {
                throw model::Cancelled();
            }
            lastError = response.error;
            bool progressed = bodyShowsProgress(response.body);
            if (!progressed) {
                try {
                    progressed = decoder(response.body, {}).progressed;
                } catch (const std::exception&) {
                    // 半截 JSON 在没有任何输出时只是可重试的连接中断
                }
            }
            if (progressed || !retryableConnection(response.code) ||
                attempt + 1 == maxAttempts) {
                throw std::runtime_error(retryMessage(attempt + 1, lastError));
            }
            waitForRetry(context, attempt, {}, false);
            continue;
        }

        if (response.status < 200 || response.status >= 300) {
            std::string message = "model returned HTTP " +
                                  std::to_string(response.status) + ": " +
                                  detail::trim(response.body.substr(0, 64 * 1024));
            lastError = message;
            if (!retryableStatus(response.status) ||
                attempt + 1 == maxAttempts) {
                throw std::runtime_error(retryMessage(attempt + 1, lastError));
            }
            auto retry = parseRetryAfter(response.retryAfter);
            waitForRetry(context, attempt, retry.first, retry.second);
            continue;
        }

        bool observed = false;
        try {
            model::Stream track = [&](const model::StreamEvent& event) {
                observed = true;
                if (streamCallback) streamCallback(event);
            };
            ParsedReply parsed = decoder(response.body, track);
            return parsed.reply;
        } catch (const model::Cancelled&) {
            throw;
        } catch (const std::exception& failure) {
            lastError = failure.what();
            bool progressed = observed || bodyShowsProgress(response.body);
            if (progressed || attempt + 1 == maxAttempts) {
                throw std::runtime_error(retryMessage(attempt + 1, lastError));
            }
            waitForRetry(context, attempt, {}, false);
        }
    }
    throw std::runtime_error(retryMessage(maxAttempts, lastError));
}

}  // namespace agentcore::openai
