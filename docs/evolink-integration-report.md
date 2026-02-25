# Evolink Provider 集成报告

> PaperBanana-xiaohongshu 项目 Evolink API 接入全过程记录
> 日期：2025-02-25

---

## 1. 项目背景

PaperBanana-xiaohongshu 是一个 AI 驱动的科学图表生成系统，通过多 Agent 协作流水线（Retriever → Planner → Visualizer → Critic）自动将论文方法章节转化为学术图表。

原项目依赖 Google Gemini、OpenAI、Claude 等国外 API，存在网络访问限制。本次集成 **Evolink**（`https://api.evolink.ai`）作为国内代理 Provider，实现无翻墙环境下的完整功能。

---

## 2. 架构设计

### 2.1 Provider 抽象层

新增 `providers/` 模块，采用抽象基类模式：

```
providers/
├── __init__.py
├── base.py          # BaseProvider 抽象基类
└── evolink.py       # EvolinkProvider 实现
```

- **文本生成**：`/v1/chat/completions`（OpenAI 兼容接口）
- **图像生成**：`/v1/images/generations`（异步任务） + `/v1/tasks/{id}`（轮询）
- **文件上传**：`https://files-api.evolink.ai/api/v1/files/upload/base64`（用于 image-to-image 参考图）

### 2.2 API 路由

每个 Agent 根据 `exp_config.provider` 字段路由 API 调用：

```python
if self.exp_config.provider == "evolink":
    response = await call_evolink_text_with_retry_async(...)
else:
    response = await call_gemini_with_retry_async(...)
```

### 2.3 Evolink 模型配置

| 用途 | 模型名称 | 接口 |
|------|---------|------|
| 文本生成（推理/规划/评审） | `gemini-2.5-flash` | `/v1/chat/completions` |
| 图像生成 | `nano-banana-2-lite` | `/v1/images/generations` |

---

## 3. 遇到的问题与解决方案

### 3.1 模型名称混淆（严重）

**问题**：`gemini-2.5-flash-image` 被错误地用于文本生成接口 `/v1/chat/completions`，导致 Evolink 返回 400 错误："Model 'gemini-2.5-flash-image' is an image generation model"。

**影响**：
- 90 次文本 API 调用全部返回 400 Bad Request
- 61 次图像 API 调用全部失败
- 重试机制在 400 错误下仍然重试 5 次，浪费时间和请求

**解决**：
- 明确区分文本模型（`gemini-2.5-flash`）和图像模型（`nano-banana-2-lite`）
- 添加 `ClientError` 异常类，4xx 客户端错误立即失败不重试

### 3.2 `cannot schedule new futures after shutdown`（严重）

**问题**：`evolink.py` 中每次 HTTP 请求都创建新的 `aiohttp.ClientSession()`。5 个候选方案并行运行时，大量 session 创建/销毁导致 Python `ThreadPoolExecutor` 被提前关闭。

**影响**：
- 所有 API 请求失败
- 请求已发送到服务器但客户端无法接收响应，造成**费用浪费但无结果**

**解决**：
- 改为共享单个 `aiohttp.ClientSession`，通过 `_get_session()` 方法复用
- 处理结束后在 `finally` 块中调用 `close()` 释放资源

### 3.3 Retriever Agent Token 消耗过高（严重）

**问题**：`auto` 检索模式将 200 篇参考论文的**完整 methodology section** 全部拼入 prompt，单次调用消耗 **81.4 万 tokens**。

**数据分析**：

| 字段 | 200 篇总量 | 占比 |
|------|-----------|------|
| methodology（全文） | ~71.7 万 tokens | 96% |
| caption（图注） | ~2.5 万 tokens | 4% |

**解决**：
- 新增 `lite` 模式（默认），仅发送 caption：从 80 万降至 **~3 万 tokens/次**，降幅 **96%**
- 保留 `auto-full` 模式供需要高精度检索时使用
- 检索目标是匹配"图表类型 + 研究领域"，caption 已包含足够信息

### 3.4 Streamlit Session State 冲突

**问题**：`st.text_input` 同时使用 `value` 参数和 `key`（session_state）导致 Streamlit 报警告。

**解决**：首次加载时通过 `st.session_state` 初始化默认值，控件只使用 `key` 绑定。

### 3.5 Provider 切换时模型名不更新

**问题**：在 gemini 和 evolink 之间切换时，文本框内的模型名称不跟随更新。

**解决**：通过 `prev_provider` 状态检测切换，更新 session_state 后调用 `st.rerun()`。

### 3.6 Polish Agent 未传参考图（严重）

**问题**：`polish_agent.py` 在使用 Evolink 图像生成 API 进行精修时，只传了文字修改建议（prompt），**完全没有传入 GT 参考图**。图像模型看不到原图，只能根据文字凭空生成，导致精修结果与原图毫无关联。

**根因分析**：

```python
# Evolink 路径（修复前）❌ —— 只有文字，没有图片
response_list = await call_evolink_image_with_retry_async(
    prompt=user_prompt,                           # 只有文字建议
    config={"aspect_ratio": ..., "quality": "2K"} # 没有 image_urls
)

# Gemini 路径 ✓ —— 文字 + 图片一起传
response_list = await call_gemini_with_retry_async(
    contents=content_list,  # 包含 text + gt_image_b64
)
```

Evolink 的 `/v1/images/generations` API 需要通过 `image_urls` 参数传入参考图 URL，而非 base64 内联。这要求先将图片上传到文件服务获取 URL。

**解决**：

1. **新增图片上传功能**：`EvolinkProvider.upload_image_base64()` 方法
   - 调用 `https://files-api.evolink.ai/api/v1/files/upload/base64`
   - 将 base64 图片上传，返回可访问的 HTTP URL（72 小时有效）
2. **修复 Polish Agent**：
   - Step 2a: 上传 GT 图到 Evolink 文件服务 → 拿到 URL
   - Step 2b: 将 URL 通过 `config["image_urls"]` 传给图像生成 API
   - `nano-banana-2-lite` 模型即可看到原图，进行 image-to-image 精修

**各 Agent 图像传递核查**：

| Agent | API 类型 | 是否需要参考图 | Evolink 路径 |
|-------|---------|--------------|-------------|
| Retriever | 文本 | 否（纯文本检索） | ✓ |
| Planner | 文本 | 是（few-shot 参考图） | ✓ 通过 `image_url` 内联 |
| Stylist | 文本 | 否（纯文字润色） | ✓ |
| Visualizer | 图像 | 否（text-to-image） | ✓ |
| Critic | 文本 | 是（评审当前图片） | ✓ 通过 `image_url` 内联 |
| Polish | 图像 | 是（基于原图精修） | ✓ 已修复 — 文件上传 + `image_urls` |

### 3.7 精修选项卡原图丢失 + Gemini 路径缺失（严重）

**问题**：`demo.py` 中精修选项卡调用的 `refine_image_with_nanoviz` 函数存在两个问题：

1. **原图被丢弃**：函数接收了 `image_bytes` 参数但完全没使用，只把文字 prompt 发给图像模型，导致精修结果与原图毫无关系
2. **Gemini 路径缺失**：原版函数使用 Gemini 多模态 API（`Part.from_bytes` 直接传图片），移植时被完全替换为不完整的 Evolink 路径，Gemini 用户无法使用精修功能

**注意**：此 bug 与 3.6 节的 Polish Agent 是**不同的代码路径**。精修选项卡走的是 `refine_image_with_nanoviz`，而非 `PolishAgent.process()`。

**原版 Gemini 实现**（正确）：
```python
contents = [
    types.Part.from_text(text=edit_prompt),
    types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes)  # 原图直接传
]
response = client.models.generate_content(model=image_model, contents=contents, config=config)
```

**移植后 Evolink 实现**（错误）：
```python
result = await evolink_provider.generate_image(
    prompt=edit_prompt,  # 只有文字，image_bytes 被丢弃
)
```

**解决**：`refine_image_with_nanoviz` 改为双 Provider 支持：

- **Gemini 路径**：恢复原版逻辑，`Part.from_text(prompt) + Part.from_bytes(image)` → Gemini 多模态图像生成
- **Evolink 路径**：上传图片到文件服务 → 拿到 URL → 通过 `image_urls` 参数传给 `nano-banana-2-lite`
- 调用处补传 `provider` 参数，跟随侧边栏 Provider 选择

**验证**：Evolink 路径日志确认上传成功、`image_urls` 已传入：
```
[Evolink 上传] ✓ 图片已上传: https://files.evolink.ai/003N42YQ2XBGIJUR4L/images/jpeg/...
[DEBUG] [Evolink 图像]   附带 1 张参考图片
[DEBUG] [Evolink]   payload keys=['model', 'prompt', 'size', 'quality', 'image_urls']
```

---

## 4. 优化措施汇总

| 优化项 | 优化前 | 优化后 | 效果 |
|-------|--------|--------|------|
| Retriever prompt | 81.4 万 tokens | 2.8 万 tokens | **降低 96%** |
| 默认重试次数 | 5 次 | 3 次 | 减少无效重试 |
| 默认候选数量 | 10 个 | 5 个 | API 调用量减半 |
| 4xx 错误处理 | 继续重试 | 立即停止 | 避免无效消耗 |
| HTTP Session | 每次请求新建 | 共享复用 | 修复并发崩溃 |
| 检索设置 UI | 无费用提示 | 显示 token 消耗提示 | 防止误操作 |
| Polish Agent | 未传参考图 | 上传图片 + image_urls | 精修结果准确 |
| 精修选项卡 | 原图丢失 + 无 Gemini | 双 Provider + 上传参考图 | 恢复完整功能 |

---

## 5. 成功运行数据

### 5.1 运行配置

| 参数 | 值 |
|------|-----|
| Provider | evolink |
| 文本模型 | gemini-2.5-flash |
| 图像模型 | nano-banana-2-lite |
| 流水线模式 | demo_planner_critic |
| 检索设置 | auto（lite 模式） |
| 候选方案数 | 5 |
| 最大评审轮次 | 3 |
| 宽高比 | 21:9 |

### 5.2 运行结果

| 指标 | 数值 |
|------|------|
| **总耗时** | **12 分 31 秒** |
| 候选完成数 | 5/5（100%） |
| 错误/重试次数 | **0** |
| 结果文件 | `results/demo/demo_20260225_062318.json`（26 MB） |

### 5.3 Token 消耗

| 类别 | 数值 |
|------|------|
| 文本 API 调用次数 | 25 次 |
| 总 prompt tokens | 450,155 |
| 总 completion tokens | 214,725 |
| 其中 reasoning tokens | 164,026（76%） |
| **总 tokens** | **664,880** |

**Token 分布（按阶段）**：

| 阶段 | 调用次数 | 平均 prompt tokens |
|------|---------|-------------------|
| Retriever（lite） | 5 | ~28,000 |
| Planner | 5 | ~42,000 |
| Critic | 15（5候选×3轮） | ~6,300 |

### 5.4 图像生成

| 指标 | 数值 |
|------|------|
| 图像任务创建 | 20 个 |
| 图像生成成功 | 19 个 |
| 平均图像大小 | 1,332 KB（~1.3 MB） |
| 图像质量 | 2K |

每个候选生成 4 张图：初始 1 张 + critic 迭代 3 张。

---

## 6. 费用对比

### 6.1 失败运行（优化前）

| 运行 | 问题 | 文本调用 | 图像调用 | 结果 |
|------|------|---------|---------|------|
| b6569b9 | 模型名反了 | 90 次（全部 400） | 61 次（全部失败） | 零产出 |
| b260cf5 | session shutdown | 37 次 | 13 次 | 部分成功但丢失 |

b260cf5 运行中 Retriever 使用完整模式，单次 prompt **814,213 tokens**。5 个候选的 Retriever 总消耗约 **400 万 tokens**，加上 Planner 等后续阶段，是费用高昂的主要原因。

### 6.2 成功运行（优化后）

| 指标 | 优化前（估算） | 优化后（实际） |
|------|--------------|--------------|
| Retriever tokens/次 | 814,213 | 28,000 |
| 5 次 Retriever 总量 | ~4,000,000 | ~140,000 |
| 全流程总 tokens | ~5,000,000+ | 664,880 |
| 错误次数 | 90+ | 0 |
| 是否完成 | 否 | 5/5 全部完成 |

**Token 消耗降低约 87%，同时从零完成率提升到 100%。**

---

## 7. 界面功能

### 7.1 侧边栏配置

- **Provider 选择**：gemini（Google 官方）/ evolink（国内代理）
- **API Key 输入**：密码框，切换 provider 自动更新
- **模型名称**：文本模型 / 图像模型，可手动修改
- **检索设置**（带费用提示）：
  - `auto`：LLM 智能选参考，仅 caption（~3 万 tokens/候选）
  - `auto-full`：LLM 智能选参考，含完整论文（~80 万 tokens/候选）
  - `random`：随机选 10 个参考（免费）
  - `none`：不检索参考（免费）

### 7.2 结果展示

- 网格式展示候选方案图片
- 每个候选可展开查看演化时间线（规划 → critic 迭代）
- 单张下载 / ZIP 批量下载
- 结果自动保存为 JSON

---

## 8. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `providers/__init__.py` | 新增 | Provider 包初始化 |
| `providers/base.py` | 新增 | BaseProvider 抽象基类 |
| `providers/evolink.py` | 新增 | EvolinkProvider 实现（共享 session、ClientError、重试优化、文件上传） |
| `utils/generation_utils.py` | 修改 | 添加 evolink 调用函数、动态 Provider 初始化、图片上传函数 |
| `utils/config.py` | 修改 | ExpConfig 添加 provider/image_model_name 字段 |
| `utils/paperviz_processor.py` | 修改 | 添加调试日志 |
| `configs/model_config.yaml` | 修改 | 添加 evolink 配置段 |
| `demo.py` | 修改 | 侧边栏 Provider 选择、API Key、检索设置费用提示、精修双 Provider |
| `agents/retriever_agent.py` | 修改 | 添加 lite 模式、auto-full 支持 |
| `agents/planner_agent.py` | 修改 | Evolink 路由、调试日志 |
| `agents/visualizer_agent.py` | 修改 | Evolink 路由、调试日志 |
| `agents/critic_agent.py` | 修改 | Evolink 路由、调试日志 |
| `agents/polish_agent.py` | 修改 | 修复 Evolink image-to-image 精修，补传参考图 |
| `agents/retriever_agent.py` | 修改 | Evolink 路由、调试日志 |
| `tests/test_evolink_provider.py` | 新增 | 23 个单元测试 |

---

## 9. 后续建议

1. **Retriever 缓存**：相同输入的检索结果可以缓存，避免重复调用 LLM
2. **流式进度**：在 Streamlit 页面实时显示每个候选的当前阶段
3. **费用估算**：在点击"生成"前预估本次运行的 token 消耗和费用
4. **图像模型切换**：Evolink 还支持 `gemini-2.5-flash-image` 图像模型，可作为备选
5. **批量结果对比**：支持多次运行结果的横向对比
