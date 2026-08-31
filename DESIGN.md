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

- `agent` 拥有认知循环、工具闭环、审批、澄清、重复失败暂停、会话快照与恢复、上下文压缩和图片保留。
- `openai` 只负责模型网络协议。Responses 是默认路径，Chat Completions 是兼容回退，连接错误与限流有限重试。
- `model` 是跨模块共享的消息、工具、流事件与 Provider 接口。

核心不假设工作区或产品身份，调用方以 Toolbox 和系统提示的身份接入。元工具只有 `request_clarification` 一个——它是循环停下来向用户要信息的通用能力；`update_plan` / `update_workflow` / `review` 是产出机制，属于 agentengine。

## 与 Agentengine 的边界

- Agentcore 提供干净的认知循环。
- Agentengine 依赖 Agentcore，在其上叠加交付工作流（goal → inspect → plan → implement → verify → review → release）、任务台账和产出元工具，并保留会话存储。
- Agentengine 的 `model` / `openai` 是 Agentcore 的类型别名，import 路径不变，下游应用无需改动。

## 设计约束

- 单进程、单 Agent，不引入后台服务。
- 公开接口：`agent`、`openai`、`model` 三个包。
- 标准库之外零依赖。
- 不引入配置中心、依赖注入、插件注册表。
