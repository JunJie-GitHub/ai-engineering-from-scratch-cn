# 结构化输出（Structured Output）：JSON、模式验证（Schema Validation）与约束解码（Constrained Decoding）

> 你的大语言模型（Large Language Model, LLM）返回的是字符串，而你的应用程序需要的是 JSON。这一鸿沟导致的生产系统崩溃，比任何模型幻觉（hallucination）都要多。结构化输出是连接自然语言与类型化数据的桥梁。做对了，你的 LLM 就会变成一个可靠的 API；做错了，你就得在凌晨三点用正则表达式（regex）去解析自由文本。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 10 阶段，课程 01-05（从零构建 LLM）
**Time:** 约 90 分钟
**Related:** 第 5 阶段 · 20（结构化输出与约束解码）涵盖了解码器（decoder）层面的理论（有限状态机/上下文无关文法（FSM/CFG）logit 处理器、Outlines、XGrammar）。本课程侧重于生产级 SDK 接口（OpenAI `response_format`、Anthropic 工具调用（tool use）、Instructor）——如果你想了解 API 底层的工作原理，请先阅读第 5 阶段 · 20。

## 学习目标

- 使用 OpenAI 和 Anthropic API 参数实现 JSON 模式（JSON-mode）与模式约束（schema-constrained）输出
- 构建 Pydantic 验证层，用于拒绝格式错误的 LLM 输出，并通过错误反馈进行重试
- 解释约束解码如何在无需后处理的情况下，在词元（token）级别强制生成有效的 JSON
- 设计鲁棒的提取提示词（prompt），可靠地将非结构化文本转换为类型化数据结构

## 问题所在

你向 LLM 提问：“从这段文本中提取产品名称、价格和库存状态。”它回答：

The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.

这个回答完全正确，但对你的应用程序来说毫无用处。你的库存系统需要 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要的是一个具有特定键、特定类型和特定值约束的 JSON 对象，而不是一句自然语言。

最直观的解决方案是：在提示词中加上“请以 JSON 格式回复”。这种方法 90% 的情况下有效。但在剩下的 10% 里，模型可能会用 Markdown 代码块包裹 JSON，或者加上“以下是 JSON：”之类的前言，又或者因为提前闭合了括号而生成语法无效的 JSON。你的 JSON 解析器会崩溃，你的数据流水线（pipeline）会中断。你不得不加上 `try/except` 和重试循环。而重试有时会产生不同的数据。现在，你在解析问题之上又叠加了一致性问题。

这并非提示词工程（prompt engineering）问题，而是解码问题。模型从左到右逐个生成词元。在每个位置，它都会从超过 10 万个候选词中挑选概率最高的下一个词元。在任意给定位置，这些候选词中的大多数都会导致生成无效的 JSON。如果模型刚刚输出了 `{"price":`，那么下一个词元必须是数字、引号（用于字符串）、`null`、`true`、`false` 或负号。任何其他内容都会导致 JSON 无效。在没有约束的情况下，模型可能会选出一个在语义上完全合理的英文单词，但在语法上却是灾难性的错误。

## 核心概念

### 结构化输出控制谱系 (Structured Output Spectrum)

结构化输出控制分为四个层级，可靠性逐级递增。

graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff

**基于提示词 (Prompt-based)**（“请以有效 JSON 格式回复”）：无强制约束。模型通常会遵守，但偶尔会失效。可靠性：约 90%。失败模式：包含 Markdown 代码块标记、前言文本、输出截断或结构错误。

**JSON 模式 (JSON mode)**：API 保证输出为有效的 JSON 格式。OpenAI 的 `response_format: { type: "json_object" }` 可启用此功能。输出可被无误解析，但可能不符合您预期的结构 (Schema)——例如包含多余键、类型错误或字段缺失。

**Schema 模式 (Schema mode)**：API 接收 JSON Schema (JSON 模式) 并保证输出与其完全匹配。到 2026 年，所有主流提供商均已原生支持此功能：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也可通过 `tool_choice="required"` 启用）、Anthropic 结合 `input_schema` 的工具使用 (Tool Use)，以及 Gemini 的 `response_schema` + `response_mime_type: "application/json"`。输出将严格包含您指定的键、类型和约束条件。

**受限解码 (Constrained decoding)**：在生成过程中的每个 Token (词元) 位置，解码器会屏蔽所有会导致无效输出的 Token。如果 Schema 要求输入数字，而模型即将输出字母，则该 Token 的概率会被设为零。模型只能生成符合有效输出的 Token。OpenAI 的结构化输出模式以及 Outlines 和 Guidance 等库底层正是采用此机制。

### JSON Schema：契约语言

JSON Schema (JSON 模式) 用于向模型（或验证层）明确指定输出必须遵循的结构。所有主流的结构化输出系统均依赖它。

{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}

该 Schema 的含义是：输出必须是一个对象，包含字符串类型的 `product`、非负数字类型的 `price`、布尔类型的 `in_stock`，以及可选的字符串数组 `categories`。任何不匹配的输出都将被拒绝。

Schema 能够处理复杂场景：嵌套对象、包含特定类型元素的数组、枚举 (Enum，将字符串限制为特定值)、模式匹配（字符串正则表达式）以及组合器（Combinators，如用于多态输出的 `oneOf`、`anyOf`、`allOf`）。

### Pydantic 模式

在 Python 中，你无需手动编写 JSON Schema。只需定义一个 Pydantic 模型，它便会自动为你生成对应的 Schema。

from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []

这将生成与上文相同的 JSON Schema。Instructor 库（以及 OpenAI 的 SDK）可直接接收 Pydantic 模型：传入模型类，即可返回经过验证的实例。若大语言模型 (LLM) 的输出不匹配，Instructor 会自动重试。

### 函数调用 / 工具使用 (Function Calling / Tool Use)

这是解决同一问题的另一种接口方式。与其直接要求模型生成 JSON，不如定义带有类型化参数的“工具”（函数）。模型会输出包含结构化参数的函数调用。OpenAI 称之为“函数调用 (Function Calling)”，Anthropic 称之为“工具使用 (Tool Use)”。最终结果一致：均返回结构化数据。

graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["User: Extract product info\nfrom this review text"] --> M["Model processes input"]
        M --> TC["Tool Call:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["Validate against\nfunction schema"]
        V --> R["Structured Result:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff

当模型需要自主选择调用哪个函数，而不仅仅是填充参数时，工具使用是更优选择。如果你有 10 种不同的提取 Schema，且模型必须根据输入内容选择正确的一种，工具使用能同时为你提供 Schema 选择与结构化输出。

### 常见失败模式

即使启用了 Schema 强制约束，结构化输出仍可能以隐蔽的方式失效。

**幻觉值 (Hallucinated values)**：输出符合 Schema，但包含虚构数据。例如原文标明价格为 $348，模型却生成 `{"price": 299.99}`。Schema 验证无法捕获此类错误——因为类型正确，仅数值有误。

**枚举混淆 (Enum confusion)**：你将某字段限制为 `["in_stock", "out_of_stock", "preorder"]`。模型却输出 `"available"`——语义正确，但不在允许的值集合内。优秀的受限解码机制可防止此类问题，而基于提示词的方法则无法做到。

**嵌套对象深度 (Nested object depth)**：深度嵌套的 Schema（4 层及以上）更容易引发错误。每一层嵌套都是模型可能丢失结构跟踪的潜在风险点。

**数组长度 (Array length)**：模型生成的数组项可能过多或过少。虽然 Schema 支持 `minItems` 和 `maxItems` 约束，但并非所有提供商都会在解码层强制执行它们。

**可选字段遗漏 (Optional field omission)**：模型可能会省略那些技术上为可选、但对你的业务场景语义上至关重要的字段。即使数据有时确实缺失，也应在 Schema 中将其设为必填（required）——以此强制模型显式输出 `null`。

## 构建项目

### 步骤 1：JSON Schema 验证器

从零开始构建一个验证器，用于检查 Python 对象是否符合 JSON Schema。该组件将在输出端运行，以验证生成结果是否合规。

import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")

### 步骤 2：Pydantic 风格模型转 Schema

构建一个轻量级的类到 Schema 转换器。定义一个 Python 类，并自动生成其对应的 JSON Schema。

class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }

### 步骤 3：受限 Token 过滤器

模拟受限解码（Constrained Decoding）过程。给定一个不完整的 JSON 字符串和一个 Schema，判断当前位置允许生成哪些 Token 类别。

def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")

### 步骤 4：信息提取流水线

将所有组件整合到一个信息提取流水线（Extraction Pipeline）中：定义 Schema，模拟大语言模型（Large Language Model, LLM）生成结构化输出，验证输出结果，并处理重试逻辑。

def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}

### 步骤 5：运行完整流水线

def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")


## 使用方法

### OpenAI 结构化输出 (Structured Outputs)

# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)

OpenAI 的结构化输出模式在内部使用了约束解码（constrained decoding）。模型生成的每个 token 都能保证输出符合 Pydantic 模式（schema）。无需重试，也无需额外验证。该约束已直接嵌入到解码过程中。

### Anthropic 工具调用 (Tool Use)

# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )

Anthropic 通过工具调用（tool use）实现结构化输出。模型会发出一个工具调用请求，其结构化参数与 `input_schema` 相匹配。最终结果相同，只是 API 接口层面有所差异。

### Instructor 库

# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )

Instructor 库封装了任意大语言模型（LLM）客户端，并添加了带验证的自动重试机制。如果首次尝试未通过验证，它会将错误信息作为上下文反馈给模型，并要求其修正输出。该方案适用于任何模型提供商，不仅限于 OpenAI。

## 交付成果

本教程将生成 `outputs/prompt-structured-extractor.md` 文件——这是一个可复用的提示词模板（prompt template），能够根据给定的模式定义从任意文本中提取结构化数据。只需向其输入 JSON Schema 和非结构化文本，它便会返回经过验证的 JSON 数据。

同时还会生成 `outputs/skill-structured-outputs.md` 文件——这是一个决策框架，帮助你根据所使用的模型提供商、可靠性要求以及模式复杂度，选择最合适的结构化输出策略。

## 练习

1. 扩展模式验证器（Schema Validator）以支持 `oneOf`（数据必须精确匹配多个模式中的某一个）。这用于处理多态输出（Polymorphic Outputs）——例如，某个字段可以是结构不同的 `Product` 或 `Service` 对象。

2. 构建一个“模式差异对比”（Schema Diff）工具，用于比较两个模式并识别破坏性变更（Breaking Changes，如移除必填字段、更改类型）与非破坏性变更（Non-breaking Changes，如添加可选字段、放宽约束）。这对于在生产环境中对提取模式（Extraction Schemas）进行版本管理至关重要。

3. 实现一个更贴近实际的约束解码模拟器（Constrained Decoding Simulator）。给定一个 JSON Schema 和包含 100 个词元（Tokens，如字母、数字、标点符号、关键字）的词汇表（Vocabulary），逐步模拟生成过程，在每个位置屏蔽无效词元。测量每一步中词汇表内有效词元所占的百分比。

4. 构建一个提取评估套件（Extraction Evaluation Suite）。创建 50 条带有手工标注 JSON 输出的产品描述。在所有 50 条数据上运行你的提取流水线（Extraction Pipeline），并测量精确匹配（Exact Match）、字段级准确率（Field-level Accuracy）以及类型合规性（Type Compliance）。找出最难正确提取的字段。

5. 为你的提取流水线添加“置信度分数”（Confidence Scores）。针对每个提取出的字段，估算模型的置信度（基于词元概率（Token Probabilities），或通过运行 3 次提取并测量结果的一致性）。将低置信度字段标记出来，以供人工复核。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| JSON 模式 (JSON mode) | “返回 JSON” | 保证输出语法合法的 JSON 的 API 标志，但不强制要求符合特定的数据结构模式 (schema) |
| 结构化输出 (Structured output) | “带类型的 JSON” | 严格匹配特定 JSON Schema 的输出，包含正确的键名、数据类型及约束条件 |
| 约束解码 (Constrained decoding) | “引导式生成” | 在生成每个词元 (token) 时，屏蔽会导致无效输出的词元 —— 保证 100% 符合模式 (schema) 要求 |
| JSON Schema | “JSON 模板” | 用于描述 JSON 数据结构、类型及约束条件的声明式语言（广泛应用于 OpenAPI、JSON Forms 等） |
| Pydantic | “Python dataclasses 增强版” | 用于定义带类型验证的数据模型的 Python 库，FastAPI 和 Instructor 等框架使用它来生成 JSON Schema |
| 函数调用 (Function calling) | “工具使用” | 大语言模型 (LLM) 输出结构化的函数调用指令（函数名 + 带类型的参数），而非自由文本 —— OpenAI 和 Anthropic 均支持该功能 |
| Instructor | “面向 LLM 的 Pydantic” | 封装 LLM 客户端的 Python 库，可直接返回通过验证的 Pydantic 实例，并在验证失败时自动重试 |
| Token 屏蔽 (Token masking) | “过滤词表” | 在生成过程中将特定词元 (token) 的概率置零，从而阻止模型输出这些词元 |
| 模式合规 (Schema compliance) | “符合结构” | 输出包含所有必填字段、正确的数据类型、符合约束的值，且不包含任何未允许的额外字段 |
| 重试循环 (Retry loop) | “重试直到成功” | 将验证错误反馈给模型并要求其修正输出 —— Instructor 会自动执行该流程，重试次数上限可配置 |

## 延伸阅读

- [OpenAI 结构化输出指南](https://platform.openai.com/docs/guides/structured-outputs) -- OpenAI API 中基于 JSON Schema 的约束解码（Constrained Decoding）官方文档
- [Willard & Louf, 2023 -- "Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702) -- Outlines 论文，阐述了如何将 JSON Schema 编译为有限状态机（Finite State Machine），以实现词元级（Token-level）约束
- [Instructor 文档](https://python.useinstructor.com/) -- 用于从任意大语言模型（Large Language Model, LLM）获取结构化输出的事实标准库，内置 Pydantic 数据验证与重试机制
- [Anthropic 工具使用指南](https://docs.anthropic.com/en/docs/tool-use) -- 介绍 Claude 如何通过工具调用（Tool Use）结合 JSON Schema 的 `input_schema` 实现结构化输出
- [JSON Schema 规范](https://json-schema.org/) -- 所有主流结构化输出系统所采用的模式语言完整规范
- [Outlines 库](https://github.com/outlines-dev/outlines) -- 开源约束生成（Constrained Generation）工具，支持将正则表达式（Regular Expression）与 JSON Schema 编译为有限状态机
- [Dong 等人, "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100) -- 当前业界领先的语法引擎；采用下推自动机（Pushdown Automaton）编译技术，以约 100 纳秒/词元的速度执行词元掩码（Token Masking）
- [Beurer-Kellner 等人, "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094) -- LMQL 论文，将约束解码抽象为一种支持类型与值约束的查询语言
- [Microsoft Guidance（框架文档）](https://github.com/guidance-ai/guidance) -- 基于模板的约束生成；作为 Outlines 与 XGrammar 的厂商无关（Vendor-agnostic）补充方案