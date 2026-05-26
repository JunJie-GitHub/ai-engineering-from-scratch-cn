# 函数调用与工具使用

> 大语言模型（Large Language Model, LLM）本身什么也做不了。它们只能生成文本，这就是它们的全部能力。它们无法查询天气、检索数据库、发送电子邮件、运行代码或读取文件。你所见过的每一个“AI 智能体（AI Agent）”，本质上都是 LLM 生成了一段 JSON 数据，指明需要调用哪个函数——随后由你的代码去实际执行。模型是大脑，工具是双手，而函数调用（Function Calling）则是连接两者的神经系统。

**类型：** 构建
**语言：** Python
**前置知识：** 第 11 阶段第 03 课（结构化输出 Structured Outputs）
**时长：** 约 75 分钟
**相关课程：** 第 11 阶段 · 14（模型上下文协议 Model Context Protocol, MCP）—— 当工具需要在不同主机间共享时，应从内联函数调用（Inline Function Calling）升级为 MCP 服务器。本课涵盖内联场景，而 MCP 课程涵盖协议场景。

## 学习目标

- 实现函数调用循环：定义工具定义（Tool Schema），解析模型生成的工具调用 JSON，执行函数并返回结果
- 设计工具定义：提供清晰的描述和类型化参数，确保模型能够可靠地调用
- 构建多轮智能体循环（Multi-turn Agent Loop）：串联多次函数调用以回答复杂查询
- 处理函数调用的边界情况：并行工具调用（Parallel Tool Calls）、错误传播（Error Propagation）以及防止无限工具循环（Infinite Tool Loops）

## 问题所在

你开发了一个聊天机器人。用户提问：“东京现在的天气怎么样？”

模型回答：“我无法获取实时天气数据，但根据当前季节，东京的气温大概在 15 摄氏度左右……”

这不过是用免责声明包装的幻觉（Hallucination）。模型根本不知道天气情况，也永远不会知道。天气每小时都在变化，而模型的训练数据可能已经是几个月前的了。

正确的答案需要调用 OpenWeatherMap API，获取当前温度，并返回真实数值。模型无法调用 API，但你的代码可以。缺失的关键环节是一个结构化协议：它允许模型表达“我需要用这些参数调用天气 API”，并让你的代码执行该调用，再将结果反馈给模型。

这就是函数调用。模型会输出结构化的 JSON，描述需要调用哪个函数以及传入什么参数。你的应用程序负责执行该函数。执行结果会被重新注入对话中，模型再基于该结果生成最终回答。

没有函数调用，大语言模型只是百科全书；有了它，它们才能成为智能体。

## 核心概念

### 函数调用循环 (Function Calling Loop)

每次工具使用交互都遵循相同的 5 步循环。

sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response

步骤 1：用户发送消息。步骤 2：模型接收该消息以及工具定义 (Tool Definitions)（描述可用函数的 JSON Schema）。步骤 3：模型不直接返回文本，而是输出一个工具调用 (Tool Call)——一个包含函数名和参数的结构化 JSON 对象。步骤 4：你的代码执行该函数并捕获结果。步骤 5：结果返回给模型，模型现在拥有了真实数据来生成最终答案。

模型从不执行任何操作。它仅决定调用什么函数以及使用什么参数。你的代码才是实际的执行者。

### 工具定义：JSON Schema 契约 (JSON Schema Contract)

每个工具都由一个 JSON Schema 定义，它告诉模型该函数的功能、需要哪些参数以及这些参数的数据类型。

{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'Tokyo' or 'San Francisco'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}

`description` 字段至关重要。模型通过阅读它们来决定何时以及如何使用工具。像“获取天气”这样模糊的描述，其工具选择效果远不如“获取某城市的当前天气。返回摄氏温度及天气状况。”描述本身就是用于工具选择的提示词 (Prompt)。

### 提供商对比 (Provider Comparison)

所有主流提供商都支持函数调用 (Function Calling)，但它们的 API 接口设计有所不同。

| 提供商 | API 参数 | 工具调用格式 | 并行调用 | 强制调用 |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | 是（每轮可多次） | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | 是（多个内容块） | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | 是 | `function_calling_config` |
| 开源权重模型 (Llama 4, Qwen3, DeepSeek-V3) | Llama 4 原生支持 `tools`；其他使用 Hermes 或 ChatML | 混合 | 取决于模型 | 基于提示词，或支持时使用 `tool_choice` |

到 2026 年，三大闭源提供商已基本收敛于几乎相同的基于 JSON Schema 的格式。Llama 4 内置了原生的 `tools` 字段，其结构与 OpenAI 一致。开源权重模型的微调版本仍存在差异——Hermes 格式（NousResearch）是第三方微调中最常用的格式。若要在不同主机间共享工具，建议优先使用 MCP（第 11 阶段 · 第 14 阶段），而非内联函数调用——因为所有客户端都可以连接同一个服务器。

### 工具选择：自动、必需、指定 (Tool Choice: Auto, Required, Specific)

你可以控制模型何时使用工具。

**自动（Auto）**（默认）：模型自行决定是调用工具还是直接回复。“2+2等于几？”——直接回复。“天气怎么样？”——调用工具。

**必需（Required）**：模型必须至少调用一个工具。当你明确知道用户意图需要工具时使用此模式。可防止模型凭空猜测而非查询真实数据。

**指定函数（Specific function）**：强制模型调用特定函数。`tool_choice={"type":"function", "function": {"name": "get_weather"}}` 可确保无论查询内容如何，都会调用天气工具。适用于路由场景——当上游逻辑已确定需要哪个工具时使用。

### 并行函数调用 (Parallel Function Calling)

GPT-4o 和 Claude 支持在单轮对话中调用多个函数。用户提问：“东京和纽约的天气怎么样？”模型会同时输出两个工具调用：

[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]

你的代码执行这两个调用（理想情况下并发执行），返回两个结果，模型随后将其综合为单一回复。这将往返次数从 2 次减少到 1 次。对于每次查询需要 5-10 次工具调用的智能体 (Agent)，并行调用可将延迟降低 60-80%。

### 结构化输出与函数调用对比 (Structured Outputs vs Function Calling)

第 03 课介绍了结构化输出 (Structured Outputs)。函数调用使用相同的 JSON Schema 机制，但目的不同。

**结构化输出**：强制模型以特定格式生成数据。输出即为最终产物。例如：从文本中提取产品信息为 `{name, price, in_stock}`。

**函数调用**：模型声明执行某项操作的意图。输出仅为中间步骤。例如：`get_weather(city="Tokyo")`——模型是在请求执行操作，而非生成最终答案。

当你需要数据提取时，使用结构化输出。当你希望模型与外部系统交互时，使用函数调用。

### 安全：不可妥协的规则 (Security: The Non-Negotiable Rules)

函数调用是你赋予大语言模型 (LLM) 的最危险能力。模型自行选择执行内容。如果你的工具集包含数据库查询，模型就会构造查询语句。如果包含 Shell 命令，模型就会编写这些命令。

**规则 1：绝不要将模型生成的 SQL 直接传递给数据库。** 模型能够且确实会生成 `DROP TABLE`、`UNION` 注入或返回所有行的查询。务必使用参数化查询。务必进行验证。务必使用操作白名单。

**规则 2：函数白名单。** 模型只能调用你明确定义的函数。切勿构建通用的“按名称执行任意函数”工具。如果你有 50 个内部函数，只暴露用户实际需要的 5 个。

**规则 3：验证参数。** 模型可能会传入类似 `"; DROP TABLE users; --"` 的城市名称。在执行前，务必根据预期的类型、范围和格式验证每个参数。

**规则 4：清理工具结果。** 如果工具返回敏感数据（API 密钥、个人身份信息 PII、内部错误），在将其发回模型前必须进行过滤。模型会原封不动地将工具结果包含在其回复中。

**规则 5：限制工具调用频率。** 处于循环中的模型可能调用工具数百次。设置上限（每次对话 10-20 次较为合理）。打破无限循环。

### 错误处理 (Error Handling)

工具会失败。API 会超时。数据库会宕机。文件可能不存在。模型需要知道工具何时失败以及失败原因。

将错误作为结构化的工具结果返回，而不是抛出异常：

{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}

模型读取该信息后，会调整参数并重试。模型擅长根据结构化的错误消息进行自我纠正。但对于空响应或泛泛的“出错了”这类错误，它们的恢复能力很差。

### MCP：模型上下文协议 (Model Context Protocol)

MCP 是 Anthropic 推出的用于工具互操作性的开放标准。MCP 提供了一种通用协议，取代了每个应用各自定义工具的做法：工具由 MCP 服务器提供，由 MCP 客户端（如 Claude Code、Cursor 或你的应用程序）消费。

一个 MCP 服务器可以向任何兼容的客户端暴露工具。Postgres MCP 服务器可为任何兼容 MCP 的智能体提供数据库访问权限。GitHub MCP 服务器可为任何智能体提供代码仓库访问权限。工具只需定义一次，即可随处使用。

MCP 之于函数调用，犹如 HTTP 之于网络通信。它标准化了传输层，从而使工具具备可移植性。

## 动手构建

### 步骤 1：定义工具注册表 (Tool Registry)

构建一个注册表，用于存储工具定义及其具体实现。每个工具均包含一个 JSON 模式 (JSON Schema) 定义（供模型识别）和一个 Python 函数（供代码执行）。

import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }

### 步骤 2：实现 5 个工具

构建计算器、天气查询、网络搜索模拟器、文件读取器以及代码运行器。

def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}

### 步骤 3：注册所有工具

def register_all_tools():
    register_tool(
        "calculator", "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"}, "precision": {"type": "integer", "description": "Decimal places in result", "default": 2}}, "required": ["expression"]},
        calculator,
    )
    register_tool(
        "get_weather", "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"}}, "required": ["city"]},
        get_weather,
    )
    register_tool(
        "web_search", "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3}}, "required": ["query"]},
        web_search,
    )
    register_tool(
        "read_file", "Read the contents of a file. Returns the file content, size, and line count.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"}}, "required": ["path"]},
        read_file,
    )
    register_tool(
        "run_code", "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "language": {"type": "string", "enum": ["python"], "description": "Programming language"}}, "required": ["code"]},
        run_code,
    )

### 步骤 4：构建函数调用循环 (Function Calling Loop)

这是系统的核心引擎。它模拟模型决定调用哪个工具的过程，执行相应工具，并将结果反馈回去。

def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}

### 步骤 5：参数验证 (Argument Validation)

构建一个验证器，在工具执行前，依据 JSON 模式 (JSON Schema) 对调用参数进行校验。

def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors

### 步骤 6：运行演示

def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")


## Use It

### OpenAI 函数调用 (Function Calling)

# from openai import OpenAI
#
# client = OpenAI()
#
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)

OpenAI 将工具调用 (Tool Calls) 作为 `response.choices[0].message.tool_calls` 返回。每次调用都包含一个 `id`，在返回结果时必须附带该 ID。模型会使用此 ID 将执行结果与对应的调用进行匹配。GPT-4o 可以在单次响应中返回多个工具调用——请遍历并执行所有调用。

### Anthropic 工具使用 (Tool Use)

# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
# )
#
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )

Anthropic 将工具调用作为类型为 `type: "tool_use"` 的内容块 (Content Blocks) 返回。工具的执行结果需放入类型为 `type: "tool_result"` 的用户消息中。请注意关键区别：Anthropic 使用 `input_schema` 来定义工具参数，而 OpenAI 使用 `parameters`。

### MCP 集成 (Integration)

# MCP servers expose tools over a standardized protocol.
# Any MCP-compatible client can discover and call these tools.
#
# Example: connecting to a Postgres MCP server
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})

MCP（Model Context Protocol）将工具的实现与调用解耦。Postgres 服务器负责处理 SQL，GitHub 服务器负责处理 API。你的智能体 (Agent) 只需发现并调用工具即可，无需为每个集成编写特定于服务提供商的代码。

## 交付上线

本课时将生成 `outputs/prompt-tool-designer.md` —— 一个用于设计工具定义（Tool Definitions）的可复用提示词模板（Prompt Template）。只需输入你期望工具执行的功能描述，它便会自动生成包含描述、类型和约束的完整 JSON Schema 定义。

同时还会生成 `outputs/skill-function-calling-patterns.md` —— 一个用于在生产环境中落地函数调用（Function Calling）的决策框架，内容涵盖工具设计、错误处理、安全性以及特定大模型服务商（Provider）的适配模式。

## 练习

1. **添加第 6 个工具：数据库查询。** 实现一个基于内存表的模拟 SQL 工具。该工具接收表名和过滤条件（而非原始 SQL 语句）。需验证表名是否在允许列表（Allowlist）中，且过滤操作符仅限于 `=`、`>`、`<`、`>=`、`<=`。将匹配的行以 JSON 格式返回。

2. **实现带错误反馈的重试机制。** 当工具调用（Tool Call）失败时（例如未找到指定城市），将错误信息回传给模型决策函数，由其自行修正参数。记录每次调用所需的尝试次数，并为每次工具调用设置最多 3 次重试的上限。

3. **构建多步智能体（Multi-step Agent）。** 某些查询需要链式调用工具：“读取配置文件并告诉我配置了哪个模型，然后在网络上搜索该模型的价格。” 实现一个循环，持续运行直到模型判定不再需要调用工具为止，并在每个决策步骤中传入累积的结果。为防止陷入无限循环，将迭代次数限制为 10 次。

4. **评估工具选择准确率。** 创建 30 个带有预期工具名称的测试查询。在所有 30 个查询上运行你的决策函数，统计其选择正确工具的百分比。找出最容易导致工具间混淆的查询。

5. **实现工具调用缓存（Tool Call Caching）。** 如果在 60 秒内使用相同参数调用同一工具，则直接返回缓存结果而非重新执行。使用以 `(tool_name, frozenset(args.items()))` 为键的字典进行存储。在包含 20 个查询的对话中测量缓存命中率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 函数调用 (Function Calling) | “工具使用” (Tool use) | 模型输出结构化的 JSON，描述需使用特定参数调用的函数——实际执行由你的代码完成，而非模型本身 |
| 工具定义 (Tool Definition) | “函数定义模式” (Function schema) | 描述工具名称、用途、参数及类型的 JSON Schema 对象——模型读取该信息以决定何时及如何使用该工具 |
| 工具选择 (Tool Choice) | “调用模式” (Calling mode) | 控制模型是必须调用工具（required）、可自行调用工具（auto），还是必须调用指定工具（named） |
| 并行调用 (Parallel Calling) | “多工具” (Multi-tool) | 模型在单轮交互中输出多个工具调用请求，从而减少往返次数——GPT-4o 和 Claude 均支持此功能 |
| 工具结果 (Tool Result) | “函数输出” (Function output) | 执行工具后返回的值，作为消息回传给模型，使其能够在回复中引用真实数据 |
| 参数校验 (Argument Validation) | “输入检查” (Input checking) | 在执行工具前，验证模型生成的参数是否符合预期的类型、取值范围及约束条件 |
| MCP (Model Context Protocol) | “工具协议” (Tool protocol) | 模型上下文协议——Anthropic 推出的开放标准，用于通过服务器暴露工具，供任何兼容的客户端发现并调用 |
| 智能体循环 (Agent Loop) | “ReAct 循环” (ReAct loop) | “模型决策调用工具 → 代码执行工具 → 结果反馈给模型”的迭代循环，直至模型获取足够信息以生成回复 |
| 工具投毒 (Tool Poisoning) | “通过工具进行提示词注入” (Prompt injection via tools) | 一种攻击手段，工具返回的结果中暗含操纵模型行为的指令——务必对所有工具输出进行安全清洗 |
| 速率限制 (Rate Limiting) | “调用预算” (Call budget) | 设置单次对话中工具调用的最大次数，以防止陷入无限循环及 API 费用失控 |

## 延伸阅读

- [OpenAI 函数调用指南](https://platform.openai.com/docs/guides/function-calling) -- 使用 GPT-4o 进行工具调用（Tool Use）的权威参考文档，涵盖并行调用（Parallel Calls）、强制调用（Forced Calling）及结构化参数（Structured Arguments）等内容。
- [Anthropic 工具使用指南](https://docs.anthropic.com/en/docs/tool-use) -- Claude 的工具调用实现方案，涵盖 `input_schema`、多工具响应（Multi-tool Responses）及 `tool_choice` 配置。
- [模型上下文协议规范](https://modelcontextprotocol.io) -- 面向 AI 应用间工具互操作性（Tool Interoperability）的开放标准，采用服务器/客户端架构（Server/Client Architecture）。
- [Schick 等人，2023 -- 《Toolformer：语言模型可以教会自己使用工具》](https://arxiv.org/abs/2302.04761) -- 训练大语言模型（Large Language Models, LLMs）自主决策何时及如何调用外部工具的基础性论文。
- [Patil 等人，2023 -- 《Gorilla：连接海量 API 的大语言模型》](https://arxiv.org/abs/2305.15334) -- 针对 1,645 个 API 对大语言模型进行微调（Fine-tuning），以实现精准调用并有效降低模型幻觉（Hallucination）。
- [伯克利函数调用排行榜](https://gorilla.cs.berkeley.edu/leaderboard.html) -- 实时基准测试（Benchmark），用于对比 GPT-4o、Claude、Gemini 及开源模型在函数调用（Function Calling）准确率上的表现。
- [Yao 等人，《ReAct：在语言模型中协同推理与行动》（ICLR 2023）](https://arxiv.org/abs/2210.03629) -- “思考-行动-观察”循环（Thought-Action-Observation Loop）构成了包裹每次工具调用的智能体（Agent）外层主循环；本课程的结尾正是第 14 阶段的起点。
- [Anthropic — 构建高效智能体（2024年12月）](https://www.anthropic.com/research/building-effective-agents) -- 基于单一工具调用原语（Primitive）构建的五种可组合模式：提示词链（Prompt Chaining）、路由（Routing）、并行化（Parallelization）、编排器-工作者（Orchestrator-Workers）以及评估器-优化器（Evaluator-Optimizer）。