# Agentcore

Agentcore 是编码 Agent 的**认知核心**：提供 Go 实现和一个行为对齐的 C++20 实现。两者都包含完整的认知循环（输入 → 推理 → 工具 → 观察 → 继续）、审批、澄清、会话检查点、上下文压缩和图像观察。它不含交付工作流、任务台账或产出机制——那些属于上层的 [Agentengine](https://github.com/momo-mnsjtxy/agentengine)，由调用方以 Toolbox 和系统提示的身份接入。

## 已有能力

- 完整认知循环：输入 → 推理 → 工具 → 观察 → 继续，直到完成、失败或取消
- 工具闭环：模型发起工具调用，核心执行并回填结果
- 同轮工具调度：默认串行保证副作用安全；Toolbox 实现 `ToolConcurrency` 并明确声明安全后，独立调用才会并发执行
- 审批：读操作自动执行，写入类动作由调用方决定是否批准，支持「记住本次决定」
- 结构化澄清：目标不明确时停止询问；连续同类失败三次自动暂停
- 完成验证：Toolbox 可选实现结构化的 `CompletionVerifier`，在模型宣称完成后检查真实环境，不通过则继续循环
- 会话检查点：原子持久化、列表、恢复
- 可恢复运行：检查点保存目标和生命周期状态，进程重启后可用 `Resume` 继续而不重复追加用户输入
- 运行预算：`RunWithOptions` 可限制轮次、工具调用、token 和时间；耗尽后进入 `paused`，可补充预算继续
- 上下文压缩：手动把旧回合压缩成有界检查点
- 图像观察：工具结果可附带截图（computer use 场景），会话只保留最近 3 张
- OpenAI Responses 流式协议（默认），Chat Completions 回退；连接错误与限流有限重试

## 使用

```go
provider := openai.New(baseURL, apiKey, model, "responses", "medium")
brain := agent.New(provider, toolbox, systemPrompt)
brain.Run(ctx, input, events, approvals)
```

`toolbox` 实现 `agent.Toolbox`（5 个必需方法），`systemPrompt` 由应用组装（可叠加 `agent.DefaultSystem()`）。

## C++ 版本

C++ 实现在 `cpp/`，命名空间是 `agentcore::model`、`agentcore::agent` 和
`agentcore::openai`。它使用 C++20、libcurl 和 JsonCpp；标准 C++ 本身没有可移植的
HTTP 客户端或 JSON 解析器，使用成熟系统库可以让业务代码保持短而可读。

```sh
cmake -S cpp -B build/agentcore-cpp
cmake --build build/agentcore-cpp
ctest --test-dir build/agentcore-cpp --output-on-failure
```

最小接入形状如下，`Provider` 和 `Toolbox` 都是显式依赖，所有权由 `shared_ptr` 表达：

```cpp
auto provider = std::make_shared<agentcore::openai::Client>(
    baseURL, apiKey, modelName, "responses", "medium");
auto brain = std::make_shared<agentcore::agent::Agent>(
    provider, toolbox, systemPrompt);
agentcore::model::Context context;
brain->run(context, input, events, approvals);
```

Responses 是默认路径；传入其他协议名会走 Chat Completions 回退。C++ 版本把
`request_clarification` 合约直接合并进模型工具列表，避免模型看不到核心拥有的澄清能力。

## 与 Agentengine 的关系

- **Agentcore**：认知核心，不含工作流/台账/产出，可独立用于任意场景。
- **Agentengine**：在 Agentcore 之上叠加交付工作流、任务台账和产出元工具，同时保留 `model` / `openai` 的同一 import 路径（type alias），下游代码无需改动。

架构边界见 [DESIGN.md](DESIGN.md)。代码遵循 HOP（Human-Oriented Programming）。

## 许可证

本项目采用 [GNU Affero General Public License v3.0（AGPL-3.0-only）](LICENSE)。
