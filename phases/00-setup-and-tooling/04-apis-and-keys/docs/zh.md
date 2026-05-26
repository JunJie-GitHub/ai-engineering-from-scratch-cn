# API 与密钥 (APIs & Keys)

> 所有 AI API 的工作方式都相同：发送请求，获取响应。细节可能不同，但模式始终如一。

**类型：** 构建 (Build)
**语言：** Python, TypeScript
**先修要求：** 第 0 阶段，第 01 课
**时长：** 约 30 分钟

## 学习目标

- 使用环境变量 (environment variables) 和 `.env` 文件安全存储 API 密钥 (API keys)
- 使用 Anthropic Python SDK 和原始 HTTP (raw HTTP) 发起大语言模型 (LLM) API 调用
- 对比基于 SDK 与原始 HTTP 的请求/响应格式 (request/response formats) 以辅助调试 (debugging)
- 识别并处理常见的 API 错误 (API errors)，包括身份验证 (authentication) 与速率限制 (rate limits)

## 问题

从第 11 阶段开始，你将调用大语言模型 API (LLM APIs)（Anthropic, OpenAI, Google）。在第 13-16 阶段，你将构建在循环中使用这些 API 的智能体 (Agents)。你需要了解 API 密钥 (API Keys) 的工作原理、如何安全地存储它们，以及如何进行首次 API 调用 (API Call)。

## 概念

sequenceDiagram
    participant C as Your Code
    participant S as API Server
    C->>S: HTTP Request (with API key)
    S->>C: HTTP Response (JSON)

每次 API 调用 (API call) 都包含：
1. 端点 (Endpoint)（URL）
2. API 密钥 (API Key)（用于身份验证）
3. 请求体 (Request Body)（即你希望发送的内容）
4. 响应体 (Response Body)（即你收到的返回结果）

## 构建它

### 步骤 1：安全存储 API 密钥 (API Keys)

切勿将 API 密钥直接写入代码中。请使用环境变量 (Environment Variables)。

export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

或者使用 `.env` 文件（请将其添加到 `.gitignore` 中）：

ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

### 步骤 2：首次 API 调用（Python）

import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is a neural network in one sentence?"}]
)

print(response.content[0].text)

### 步骤 3：首次 API 调用（TypeScript）

import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 256,
  messages: [{ role: "user", content: "What is a neural network in one sentence?" }],
});

console.log(response.content[0].text);

### 步骤 4：原始 HTTP 请求（不使用 SDK）

import os
import urllib.request
import json

url = "https://api.anthropic.com/v1/messages"
headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
    "anthropic-version": "2023-06-01",
}
body = json.dumps({
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "What is a neural network in one sentence?"}],
}).encode()

req = urllib.request.Request(url, data=body, headers=headers, method="POST")
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(result["content"][0]["text"])

这正是软件开发工具包 (SDK) 在底层所执行的操作。理解原始 HTTP 调用有助于调试 (Debugging)。

## 使用方法

本课程需要用到以下服务：

| API | 使用时机 | 免费套餐 (Free tier) |
|-----|-----------------|-----------|
| Anthropic (Claude) | 第 11-16 阶段（智能体 (agents)、工具 (tools)） | 注册即送 5 美元额度 |
| OpenAI | 第 11 阶段（对比） | 注册即送 5 美元额度 |
| Hugging Face | 第 4-10 阶段（模型 (models)、数据集 (datasets)） | 免费 |

你目前无需全部配置。等到课程需要时再进行设置即可。

## 发布

本节将生成：
- `outputs/prompt-api-troubleshooter.md` - 诊断常见的 API 错误

## 练习

1. 获取 Anthropic API 密钥 (API key) 并进行首次 API 调用 (API call)
2. 尝试原始 HTTP (raw HTTP) 版本，并将响应格式与 SDK 版本 (SDK version) 进行对比
3. 故意使用错误的 API 密钥，并查看错误信息 (error message)

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| API 密钥 (API key) | “API 的密码” | 用于标识账户身份并授权请求的唯一字符串 |
| 速率限制 (Rate limit) | “被限流了” | 每分钟/小时允许的最大请求数，旨在防止滥用并确保公平使用 |
| 词元 (Token) | “一个词”（在 API 语境中） | 计费单位：输入和输出的词元会分别统计并独立计费 |
| 流式输出 (Streaming) | “实时响应” | 逐字获取响应内容，而无需等待完整响应生成完毕 |