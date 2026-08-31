package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/momo-mnsjtxy/agentcore/model"
)

func TestCompleteRetriesRateLimitsForBothProtocols(t *testing.T) {
	for _, protocol := range []string{"responses", "chat"} {
		t.Run(protocol, func(t *testing.T) {
			var requests atomic.Int32
			client := New("https://api.example.test/v1", "", "test-model", protocol, "none")
			client.http = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				if requests.Add(1) == 1 {
					return responseWith(request, http.StatusTooManyRequests, "rate limited", http.Header{"Retry-After": []string{"1"}}), nil
				}
				return streamResponse(request, successfulStream(protocol, "ok"), nil), nil
			})}
			var waits []time.Duration
			client.wait = func(_ context.Context, delay time.Duration) error {
				waits = append(waits, delay)
				return nil
			}
			client.jitter = func(time.Duration) time.Duration { return 0 }

			reply, err := client.Complete(context.Background(), nil, nil, func(model.StreamEvent) {})
			if err != nil {
				t.Fatal(err)
			}
			if reply.Content != "ok" || requests.Load() != 2 {
				t.Fatalf("reply=%q requests=%d", reply.Content, requests.Load())
			}
			if len(waits) != 1 || waits[0] != time.Second {
				t.Fatalf("Retry-After was not respected: %#v", waits)
			}
		})
	}
}

func TestCompleteChatRequestsStreamUsage(t *testing.T) {
	client := New("https://api.example.test/v1", "", "test-model", "chat", "none")
	client.http = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		var payload map[string]any
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		options, ok := payload["stream_options"].(map[string]any)
		if !ok || options["include_usage"] != true {
			t.Fatalf("chat request did not ask for usage: %#v", payload)
		}
		return streamResponse(request, successfulStream("chat", "ok"), nil), nil
	})}

	if _, err := client.Complete(context.Background(), nil, nil, func(model.StreamEvent) {}); err != nil {
		t.Fatal(err)
	}
}

func TestCompleteUsesJitteredBackoffWithoutRetryAfter(t *testing.T) {
	var requests atomic.Int32
	client := New("https://api.example.test/v1", "", "test-model", "responses", "none")
	client.http = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if requests.Add(1) == 1 {
			return responseWith(request, http.StatusServiceUnavailable, "unavailable", nil), nil
		}
		return streamResponse(request, successfulStream("responses", "recovered"), nil), nil
	})}
	client.jitter = func(limit time.Duration) time.Duration {
		if limit != 125*time.Millisecond {
			t.Fatalf("unexpected jitter limit: %s", limit)
		}
		return 25 * time.Millisecond
	}
	var waited time.Duration
	client.wait = func(_ context.Context, delay time.Duration) error {
		waited = delay
		return nil
	}

	if _, err := client.Complete(context.Background(), nil, nil, func(model.StreamEvent) {}); err != nil {
		t.Fatal(err)
	}
	if waited != 275*time.Millisecond {
		t.Fatalf("unexpected backoff: %s", waited)
	}
}

func TestCompleteCapsRetriesAndDoesNotRetryClientErrors(t *testing.T) {
	t.Run("retry cap", func(t *testing.T) {
		var requests atomic.Int32
		client := New("https://api.example.test/v1", "", "test-model", "responses", "none")
		client.http = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests.Add(1)
			return responseWith(request, http.StatusBadGateway, "unavailable", nil), nil
		})}
		client.wait = func(context.Context, time.Duration) error { return nil }
		client.jitter = func(time.Duration) time.Duration { return 0 }
		_, err := client.Complete(context.Background(), nil, nil, func(model.StreamEvent) {})
		if err == nil || !strings.Contains(err.Error(), "after 3 attempts") || requests.Load() != maxRequestAttempts {
			t.Fatalf("err=%v requests=%d", err, requests.Load())
		}
	})

	t.Run("client error", func(t *testing.T) {
		var requests atomic.Int32
		client := New("https://api.example.test/v1", "", "test-model", "responses", "none")
		client.http = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests.Add(1)
			return responseWith(request, http.StatusBadRequest, "bad request", nil), nil
		})}
		client.wait = func(context.Context, time.Duration) error {
			t.Fatal("non-retryable error entered backoff")
			return nil
		}
		_, err := client.Complete(context.Background(), nil, nil, func(model.StreamEvent) {})
		if err == nil || requests.Load() != 1 {
			t.Fatalf("err=%v requests=%d", err, requests.Load())
		}
	})
}

func TestCompleteRetriesConnectionFailureBeforeOutput(t *testing.T) {
	var requests atomic.Int32
	client := New("https://api.example.test/v1", "", "test-model", "responses", "none")
	client.http = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if requests.Add(1) == 1 {
			return nil, &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection reset")}
		}
		return streamResponse(request, successfulStream("responses", "recovered"), nil), nil
	})}
	client.wait = func(context.Context, time.Duration) error { return nil }
	client.jitter = func(time.Duration) time.Duration { return 0 }

	reply, err := client.Complete(context.Background(), nil, nil, func(model.StreamEvent) {})
	if err != nil || reply.Content != "recovered" || requests.Load() != 2 {
		t.Fatalf("reply=%q err=%v requests=%d", reply.Content, err, requests.Load())
	}
}

func TestCompleteRetriesStreamFailureBeforeFirstEvent(t *testing.T) {
	var requests atomic.Int32
	client := New("https://api.example.test/v1", "", "test-model", "responses", "none")
	client.http = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if requests.Add(1) == 1 {
			return streamResponse(request, "", io.ErrUnexpectedEOF), nil
		}
		return streamResponse(request, successfulStream("responses", "recovered"), nil), nil
	})}
	client.wait = func(context.Context, time.Duration) error { return nil }
	client.jitter = func(time.Duration) time.Duration { return 0 }

	reply, err := client.Complete(context.Background(), nil, nil, func(model.StreamEvent) {})
	if err != nil || reply.Content != "recovered" || requests.Load() != 2 {
		t.Fatalf("reply=%q err=%v requests=%d", reply.Content, err, requests.Load())
	}
}

func TestCompleteDoesNotReplayAfterStreamOutput(t *testing.T) {
	var requests atomic.Int32
	partial := `data: {"type":"response.output_text.delta","delta":"partial"}` + "\n\n"
	client := New("https://api.example.test/v1", "", "test-model", "responses", "none")
	client.http = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		requests.Add(1)
		return streamResponse(request, partial, io.ErrUnexpectedEOF), nil
	})}
	client.wait = func(context.Context, time.Duration) error {
		t.Fatal("stream with output entered backoff")
		return nil
	}

	var output strings.Builder
	_, err := client.Complete(context.Background(), nil, nil, func(event model.StreamEvent) {
		if event.Kind == model.OutputDelta {
			output.WriteString(event.Text)
		}
	})
	if err == nil || output.String() != "partial" || requests.Load() != 1 {
		t.Fatalf("output=%q err=%v requests=%d", output.String(), err, requests.Load())
	}
}

func TestCompleteDoesNotReplayAfterToolCallOutput(t *testing.T) {
	var requests atomic.Int32
	partial := `data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"run","arguments":""}}` + "\n\n"
	client := New("https://api.example.test/v1", "", "test-model", "responses", "none")
	client.http = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		requests.Add(1)
		return streamResponse(request, partial, io.ErrUnexpectedEOF), nil
	})}
	client.wait = func(context.Context, time.Duration) error {
		t.Fatal("stream with tool output entered backoff")
		return nil
	}

	_, err := client.Complete(context.Background(), nil, nil, func(model.StreamEvent) {})
	if err == nil || requests.Load() != 1 {
		t.Fatalf("err=%v requests=%d", err, requests.Load())
	}
}

func TestRetryWaitHonorsCancellation(t *testing.T) {
	client := New("https://api.example.test/v1", "", "test-model", "responses", "none")
	client.http = &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return responseWith(request, http.StatusServiceUnavailable, "busy", nil), nil
	})}
	client.wait = func(context.Context, time.Duration) error { return context.Canceled }
	_, err := client.Complete(context.Background(), nil, nil, func(model.StreamEvent) {})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, time.August, 15, 12, 0, 0, 0, time.UTC)
	if delay, ok := parseRetryAfter("2", now); !ok || delay != 2*time.Second {
		t.Fatalf("numeric Retry-After=%s ok=%v", delay, ok)
	}
	date := now.Add(3 * time.Second).Format(http.TimeFormat)
	if delay, ok := parseRetryAfter(date, now); !ok || delay != 3*time.Second {
		t.Fatalf("date Retry-After=%s ok=%v", delay, ok)
	}
	if delay, ok := parseRetryAfter(now.Add(-time.Second).Format(http.TimeFormat), now); !ok || delay != 0 {
		t.Fatalf("past Retry-After=%s ok=%v", delay, ok)
	}
}

func successfulStream(protocol, text string) string {
	if protocol == "chat" {
		stream := `data: {"choices":[{"delta":{"content":"` + text + `"}}]}` + "\n\n" + "data: [DONE]\n\n"
		return stream
	}
	stream := `data: {"type":"response.output_text.delta","delta":"` + text + `"}` + "\n\n" +
		`data: {"type":"response.completed","response":{"id":"resp_test"}}` + "\n\n" + "data: [DONE]\n\n"
	return stream
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type errorAfterBody struct {
	reader *strings.Reader
	err    error
}

func (body *errorAfterBody) Read(buffer []byte) (int, error) {
	if body.reader.Len() > 0 {
		return body.reader.Read(buffer)
	}
	return 0, body.err
}

func (*errorAfterBody) Close() error { return nil }

func streamResponse(request *http.Request, content string, finalError error) *http.Response {
	body := io.ReadCloser(io.NopCloser(strings.NewReader(content)))
	if finalError != nil {
		body = &errorAfterBody{reader: strings.NewReader(content), err: finalError}
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     make(http.Header),
		Body:       body,
		Request:    request,
	}
}

func responseWith(request *http.Request, status int, content string, header http.Header) *http.Response {
	if header == nil {
		header = make(http.Header)
	}
	return &http.Response{
		StatusCode: status,
		Status:     http.StatusText(status),
		Header:     header,
		Body:       io.NopCloser(strings.NewReader(content)),
		Request:    request,
	}
}
