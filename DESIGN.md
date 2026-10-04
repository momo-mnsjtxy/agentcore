# Agentcore 设计

Agentcore（认知核心）的目标是提供一个极简、可独立演进的 Agent 认知循环，不含任何"产出"机制——没有交付工作流、没有任务台账、没有计划/审查元工具。这是它与 Agentengine 的分界。

## 主链

```text
用户输入
  -> Agent 组装当前会话
  -> Responses 或 Chat 流式推理
  -> 回答 / 推理 / 用量事件交给调用方
  -> 模型请求工具或 request_clarification
  -> 必要时等待用户审批
  -> 执行工具并把观察结果放回会话
  -> 继续推理，直到回答完成、失败或取消
```

完整认知循环在 `agent/agent.go`，应当可以从上到下直接阅读。

## 模块边界

- `agent` 拥有认知循环、工具闭环、审批、澄清、重复失败暂停、会话快照与恢复、预算暂停、完成验证协议和图片保留。
- `openai` 只负责模型网络协议。Responses 是默认路径，Chat Completions 是兼容回退，连接错误与限流有限重试。
- `model` 是跨模块共享的消息、工具、流事件与 Provider 接口。

核心不假设工作区或产品身份，调用方以 Toolbox 和系统提示的身份接入。元工具只有 `request_clarification` 一个——它是循环停下来向用户要信息的通用能力；`update_plan` / `update_workflow` / `review` 是产出机制，属于 agentengine。

工具默认串行执行，以保护未知的外部副作用。只有实现 `ToolConcurrency` 并明确声明调用
安全的 Toolbox 才会获得同轮并行调度；并行安全性由调用方负责，核心不猜测工具语义。
`CompletionVerifier` 同样只定义结构化验证协议，真实的测试、文件、数据库或业务状态检查由
应用实现。

## 与 Agentengine 的边界

- Agentcore 提供干净的认知循环。
- Agentengine 依赖 Agentcore，在其上叠加交付工作流（goal → inspect → plan → implement → verify → review → release）、任务台账和产出元工具，并保留会话存储。
- Agentengine 的 `model` / `openai` 是 Agentcore 的类型别名，import 路径不变，下游应用无需改动。

## 设计约束

- 单进程、单 Agent，不引入后台服务。
- 公开接口：`agent`、`openai`、`model` 三个包。
- 标准库之外零依赖。
- 不引入配置中心、依赖注入、插件注册表。

## C++ 实现

`cpp/` 保持相同的三个业务边界和认知行为，使用 C++20。Go 版本继续保持标准库零依赖；
C++ 版本明确依赖系统 libcurl 与 JsonCpp，因为语言标准没有 HTTP 和 JSON 能力。依赖放在
CMake 的目标上，不藏在全局状态或运行时注册表里。

C++ 代码按 HOP 组织：主循环在 `agent/agent.cpp`，会话状态、压缩和澄清分别在相邻的
业务文件中；OpenAI 的请求、Responses、Chat 和传输重试各自保持一条可读主线。必要的
内存和网络边界仍做显式检查，HOP 的“信任数据”不用于绕过 C++ 的未定义行为风险。
