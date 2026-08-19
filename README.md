# ESP32-S3 中文古典诗词大模型

在 ESP32-S3 上**完全离线**运行一个自训练的中文古典诗词 Transformer：不联网、不调 API，权重常驻 PSRAM，逐字自回归生成，并在 2.0 寸彩屏上以打字机效果流式呈现。

本项目 fork 自 [DaveBben/esp32-llm](https://github.com/DaveBben/esp32-llm)（在 ESP32 上跑 TinyStories 的英文演示），推理内核源自 [karpathy/llama2.c](https://github.com/karpathy/llama2.c)。与上游的区别是：**模型不再是下载来的英文 checkpoint，而是本仓库 `tools/` 全套脚本从全唐诗语料自己训练出来的中文诗词模型**。

> 注：仓库里的 `ESP32_LLM.jpg` 与 `llm_output.gif` 是上游英文 stories 版本的演示素材，尚未替换为当前中文彩屏版本的实拍。

---

## 模型规格

从零训练的紧凑 Llama-2 架构：

| 项目 | 取值 |
| :--- | :--- |
| 隐藏层维度 `dim` | 96 |
| 前馈维度 `hidden_dim` | 256 |
| 层数 `n_layers` | 5 |
| 注意力头 `n_heads` / `n_kv_heads` | 6 / 6（head_size = 16） |
| 词表 `vocab_size` | 2048（字符级，覆盖语料 96.4% 字次） |
| 上下文 `seq_len` | 128 |
| 参数量 | 750,624（词嵌入与输出层权重共享） |
| 权重文件 `data/poem_model.bin` | 约 2.87 MiB（float32） |
| 分词器 `data/poem_tok.bin` | 约 22 KB |

语料为 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) 全唐诗，经繁简转换与清洗后保留约 1.08 万首（详见 `tools/download_corpus.py`）。

## 提示词格式（重要）

模型是**按主题与体裁条件化**训练的，不是"给上句续下句"。提示词必须是这个格式：

```
主题：明月 体裁：五绝\n
体裁：七绝\n            # 只指定体裁
主题：故乡\n            # 只指定主题
```

可用体裁标签：`五绝` `七绝` `五律` `七律` `五言` `七言`。
主题词取自 `tools/dataset.py` 的 `COMMON_THEMES`（明月、春风、边塞、相思、江南……）及单字主题（月、风、花、雪……）。

给一句"床前明月光，"这样的裸首句属于分布外输入，效果会明显变差。

板载提示词轮播列表硬编码在 [`main/main.c`](main/main.c) 的 `prompts[]` 数组里，采样参数（`temperature`、`topp`、`steps`）同样在 `app_main()` 中，微控制器上没有命令行可传参。

---

## 硬件

- **主控**：ESP32-S3，需 **8MB Octal (OPI) PSRAM**（模型权重 2.87MB + KV Cache + 激活值）与 16MB Flash。
- **默认板型**：Waveshare ESP32-S3-Touch-LCD-2（2.0 寸 240x320 ST7789 SPI 彩屏，40MHz，背光 GPIO 1）。

通过 `idf.py menuconfig` → `Board Model` 可切换三种目标：

| 选项 | 说明 |
| :--- | :--- |
| `Waveshare ESP32-S3-Touch-LCD-2` | 默认，彩屏中文排版 + 逐字动画 |
| `Generic ESP32-S3 with SSD1306 I2C OLED` | 外接 128x64 单色 OLED |
| `Headless` | 无屏，仅串口输出 |

---

## 构建与烧录

需要 [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html#installation) 工具链（目标 `esp32s3`，组件版本锁定见 `dependencies.lock`）。

Windows PowerShell 下先加载环境：

```powershell
. C:\Espressif\tools\Microsoft.v6.0.1.PowerShell_profile.ps1
```

然后：

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p COMx flash monitor      # Linux/macOS 换成 /dev/ttyUSB0 之类
```

`data/` 目录会被打包为 SPIFFS 镜像烧入独立分区（4MB，见 `partitions.csv`），`idf.py flash` 会连同权重一起烧录。

---

## 重新训练

训练工具链在 `tools/`，依赖 PyTorch（仓库内 `.venv/` 已备好环境，含 Intel XPU 构建）。

```bash
# 1. 下载并清洗全唐诗语料 -> tools/poetry_corpus.json
python tools/download_corpus.py

# 2. 训练（内部自动构建词表并导出二进制权重）
python tools/train_zh_poem.py

# 3. 重新生成屏幕字库（必须！见下方警告）
python tools/build_zh_font.py

# 4. 重新编译烧录
idf.py build && idf.py -p COMx flash
```

`train_zh_poem.py` 会依次产出：
`tools/vocab.json` → `data/poem_tok.bin` → `checkpoints/poem_model.pt`（可续训，不在 SPIFFS 镜像内）→ `data/poem_model.bin`（固件用）。

> ⚠️ **字库与词表强耦合**：`main/zh_font_16x16.c` 是**按 token id 直接索引**的点阵表。只要词表顺序变了（改语料、改 `dataset.py`、改 `VOCAB_SIZE` 都会变），就必须重跑 `tools/build_zh_font.py`，否则屏幕会静默显示成错字——因为那只是一次数组查表，不会报任何错。串口输出走 UTF-8 解码，不受影响。

### 训练配置

超参数在 [`tools/train_zh_poem.py`](tools/train_zh_poem.py) 顶部：

- 6000 步 × batch 32 × 128 token，AdamW + 余弦退火（3e-3 → 1e-4，200 步 warmup）
- 样本按 `<s>…</s>` 拼接成连续 token 流后随机切窗（sequence packing），避免 pad 浪费算力
- 训练/验证集**按诗划分**（同一首诗会展开成多条条件化样本，按样本划分会导致泄漏），每 500 步报验证 loss
- **保留验证 loss 最低的权重**并在导出前回滚——本模型约 2500 步后开始过拟合，最后一步并非最好的一步
- 固定随机种子 `SEED = 1337`，结果可复现
- 自动适配 CUDA / Intel XPU / CPU，三者均走 bfloat16 AMP

当前发布模型的验证 loss 曲线（Arc 130T，6000 步）：

| 步数 | 500 | 1500 | **2500** | 4000 | 6000 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 验证 loss | 4.85 | 4.46 | **4.3533（最优，已导出）** | 4.50 | 4.63 |

训练耗时：Arc 130T 约 8.2 分钟，CPU（7 线程）约 15.2 分钟。

> ⚠️ **不要为了"停在最优点"而把 `MAX_STEPS` 改小**。该值同时决定余弦退火的衰减跨度：改成 3000 后 LR 衰减更快，最优点反而更差（3000 步计划最低 4.4259，6000 步计划最低 4.3865，两者都出现在第 2500 步附近）。既然已有最优权重回滚机制兜住过拟合尾段，保持较长、衰减较慢的计划严格更优，代价只是墙上时间。

### Intel Arc / XPU 加速

若机器有 Intel Arc 独显或核显（如 Core Ultra 的 Arc 130T），**默认从 PyPI 装的 `torch` 是 CPU-only 版本**（版本号形如 `2.13.0+cpu`），`torch.xpu.is_available()` 恒为 `False`。需换成 XPU 构建：

```bash
pip install --index-url https://download.pytorch.org/whl/xpu torch
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
# 期望输出：2.13.0+xpu True
```

前提是已安装较新的 Intel 显卡驱动。训练脚本会自动优先选用 XPU，无需改代码。

> PyTorch 2.5 起 Intel GPU 支持已并入上游，**不再需要 `intel_extension_for_pytorch`**；脚本中的 IPEX 调用现已限定为仅作用于 CPU 路径。

### PC 端韵律模拟器

```bash
python tools/generate_zh_poem.py
```

在 PC 上加载 `data/poem_model.bin` 复现端侧生成，并叠加**中华通韵约束解码**：开局先选定韵部，强制第 2、4 句末字落在同一韵部（logits 掩码）。需要 `pypinyin`。

注意这是**中华通韵**（现代普通话韵母），不是平水韵——平水韵需要入声与历史韵部数据，拼音无法还原。

---

## 性能优化

固件侧相对 llama2.c 的改动：

1. **双核并发**：`matmul()` 与 `forward()` 内的注意力循环对半拆分，Core 0/1 各跑一个绑核 FreeRTOS 任务，用信号量与 `xEventGroupSync` 屏障同步。
2. **ESP-DSP SIMD**：矩阵乘调用 [`dsps_dotprod_f32_aes3`](https://github.com/espressif/esp-dsp/tree/master/modules/dotprod/float)，利用 ESP32-S3 的 SIMD 指令。
3. **时钟与缓存**：CPU 240MHz、PSRAM 80MHz OPI、指令缓存放大到 32KB（`sdkconfig`）。
4. **内存布局**：`linker.lf` 把 `libmain.a` 放进 IRAM/DRAM 提速，但 64KB 字库点阵表显式排除回 Flash，把 DRAM 留给模型与 KV Cache。

> 实测约 **17.6–18.5 tok/s**（Waveshare ESP32-S3-Touch-LCD-2，词表 2048）。上一版词表 1024 时约 36.6 tok/s——扩表后输出投影层参数翻倍，速率大致减半，符合预期。

---

## 目录结构

```
main/
  main.c              app_main：初始化屏幕/SPIFFS，构建模型，轮播提示词生成
  llm.c / llm.h       llama2.c 移植：权重加载、BPE 分词、前向传播、采样
  display_driver.*    屏幕硬件抽象（Waveshare ST7789 / SSD1306 / Headless）
  zh_font_16x16.*     自动生成的 16x16 中文点阵字库（勿手改）
  llama.h             启动画面 XBM 位图
tools/
  download_corpus.py    下载清洗全唐诗
  dataset.py            主题/体裁条件化样本构造 + 训练验证集划分
  build_zh_tokenizer.py 字符级词表 -> vocab.json / poem_tok.bin
  model.py              PyTorch Llama-2 定义
  train_zh_poem.py      训练主脚本
  export_zh_model.py    导出 llama2.c 二进制格式
  build_zh_font.py      按 vocab.json 生成屏幕点阵字库
  generate_zh_poem.py   PC 端押韵约束推理模拟器
data/                   SPIFFS 镜像内容（模型权重 + 分词器）
docs/TECHNICAL_DESIGN.md  技术方案与实现说明
```

---

## 致谢与许可

- 推理内核：[karpathy/llama2.c](https://github.com/karpathy/llama2.c)（MIT）
- ESP32 移植起点：[DaveBben/esp32-llm](https://github.com/DaveBben/esp32-llm)
- 语料：[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT）
- 字库点阵由本机系统字体（SimHei）渲染生成，分发前请确认字体授权。
