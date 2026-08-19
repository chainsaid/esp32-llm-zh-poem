# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在处理本仓库代码时提供指引与参考。

## 项目概述

本项目在 ESP32-S3 上完全离线运行一个**自训练的中文古典诗词 Transformer**，逐 token 生成并在彩屏上流式显示，同时上报生成速率（tokens/sec）。

推理内核移植自 [llama2.c](https://github.com/karpathy/llama2.c)，仓库最初 fork 自跑英文 TinyStories 的 [esp32-llm](https://github.com/DaveBben/esp32-llm)。**但模型已不再是下载来的 stories260K checkpoint** —— 现在的权重由本仓库 `tools/` 下的完整工具链从全唐诗语料训练得出。

模型规格（与代码严格一致，改动时请同步更新本表）：

| 项目 | 取值 |
| :--- | :--- |
| `dim` / `hidden_dim` | 96 / 256 |
| `n_layers` | 5 |
| `n_heads` / `n_kv_heads` | 6 / 6（head_size = 16） |
| `vocab_size` | 2048（字符级） |
| `seq_len` | 128 |
| 参数量 | 750,624（词嵌入与输出层权重共享） |
| `data/poem_model.bin` | 约 2.87 MiB（float32） |
| `data/poem_tok.bin` | 约 22 KB |

目标硬件为带 **8MB Octal (OPI) PSRAM** 的 ESP32-S3。默认板型是 Waveshare ESP32-S3-Touch-LCD-2（240x320 ST7789 SPI 彩屏），可在 `idf.py menuconfig` → `Board Model` 切换到通用 SSD1306 I2C OLED 或无屏 Headless 模式。

## 提示词格式

模型按**主题 + 体裁条件化**训练，不是首句续写。有效提示词形如：

```
主题：明月 体裁：五绝\n
体裁：七绝\n
主题：故乡\n
```

体裁标签仅这六种：`五绝` `七绝` `五律` `七律` `五言` `七言`。裸首句（如 `床前明月光，`）属分布外输入。

提示词轮播列表与采样参数（`temperature`、`topp`、`steps`、`rng_seed`）硬编码在 `main/main.c` 的 `app_main()` 中——微控制器没有命令行接口，调整生成行为请直接改这里。

## 编译与烧录命令

依赖 [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html#installation) 工具链（目标芯片 `esp32s3`，组件管理器锁定版本见 `dependencies.lock`）。

Windows PowerShell 下执行 `idf.py` 前需先加载环境变量：

```powershell
. C:\Espressif\tools\Microsoft.v6.0.1.PowerShell_profile.ps1
```

常用命令：

```bash
idf.py build
idf.py -p COMx flash
idf.py -p COMx monitor
```

（Linux/macOS 下将 `COMx` 换为 `/dev/ttyUSB0` 等设备路径）

本项目没有独立的单元测试或 lint 配置。固件正确性通过编译并烧录至真实硬件验证（串口监视器可看生成文本与 tok/s）。

`data/` 目录（`poem_model.bin` 权重 + `poem_tok.bin` 分词器）会被打包为 SPIFFS 镜像烧入独立分区（见 `main/CMakeLists.txt` 的 `spiffs_create_partition_image` 与 `partitions.csv` 中 4MB 的 `data` 分区）。修改 `data/` 内容后，`idf.py flash` 会随常规构建流程重新生成并烧录该分区。

## 训练工具链（tools/）

Python 环境在仓库内 `.venv/`（torch XPU 版、Pillow、pypinyin、zhconv）。完整重训流程：

```bash
python tools/download_corpus.py      # 1. 下载清洗全唐诗 -> tools/poetry_corpus.json
python tools/train_zh_poem.py        # 2. 训练（内部自动建词表 + 导出权重）
python tools/build_zh_font.py        # 3. 重新生成屏幕点阵字库（见下方警告）
idf.py build && idf.py -p COMx flash # 4. 重新烧录
```

各脚本职责：

- **`download_corpus.py`**：从 chinese-poetry 仓库拉取全唐诗，繁转简、过滤残缺与超长诗，输出约 1.08 万首到 `tools/poetry_corpus.json`。
- **`dataset.py`**：把每首诗展开为主题/体裁条件化样本（`主题：X 体裁：Y\n{正文}`、仅体裁、仅主题、无条件各一份）。同时提供 `get_poem_split_ids()` 做**按诗**的训练/验证集划分——一首诗会产生多条样本，按样本划分会造成泄漏。
- **`build_zh_tokenizer.py`**：按语料字频构建字符级词表（`VOCAB_SIZE = 2048`），输出 `tools/vocab.json` 与 llama2.c 格式的 `data/poem_tok.bin`。ID 布局：0–3 特殊 token（`<unk>` `<s>` `</s>` `\n`），4–23 标点与数字（含 ID 13 空格），24 起为按频次排序的汉字。
- **`model.py`**：PyTorch Llama-2 结构定义（RMSNorm / RoPE / SwiGLU / 权重共享）。
- **`train_zh_poem.py`**：训练主脚本。详见下节。
- **`export_zh_model.py`**：导出 llama2.c 二进制格式（28 字节头 + 扁平 float32 权重 + RoPE 占位）。
- **`build_zh_font.py`**：按 `vocab.json` 顺序渲染 16x16 点阵，生成 `main/zh_font_16x16.c/.h`。
- **`generate_zh_poem.py`**：PC 端推理模拟器，叠加中华通韵约束解码（开局选定韵部，强制第 2、4 句末字同韵）。

### ⚠️ 字库与词表强耦合

`main/zh_font_16x16.c` 是**按 token id 直接索引**的点阵数组，只对生成它的那份 `vocab.json` 有效。词表顺序由语料字频决定，因此**改语料、改 `dataset.py`、改 `VOCAB_SIZE` 都会让它失效**。

失效表现是**静默的**：`display_driver_append_token()` 只做一次数组查表，id 越界直接 `return`，不越界就画出错字，全程无任何报错。串口输出走 `decode()` 的 UTF-8 路径，不受影响——所以"串口正常但屏幕乱码"就是这个原因。

**任何时候重建了词表，都必须重跑 `python tools/build_zh_font.py`。**

### 训练脚本要点

- 超参数在 `train_zh_poem.py` 顶部：6000 步 × batch 32 × seq 128，AdamW + 余弦退火（3e-3 → 1e-4，200 步 warmup），`SEED = 1337` 保证可复现。
- **验证 loss 约 2500 步触底后回升**（6000 步末尾 4.62，2500 步 4.39）。脚本保留验证 loss 最低的权重并在导出前回滚，所以尾段过拟合不影响产物。
- **不要把 `MAX_STEPS` 改小来"停在最优点"**：它同时是余弦退火的衰减跨度，改成 3000 会让 LR 衰减更快、最优点更差（4.4259 vs 4.3865）。
- **Sequence packing**：样本用 `<s>…</s>` 拼成连续 token 流后随机切窗，而非 pad 到 `seq_len`。诗词样本平均仅约 63 token，pad 方案会浪费约一半算力。
- 每 500 步在验证集上报 loss（`EVAL_ITERS = 40` 个 batch 平均）。
- 设备自适应：CUDA / Intel XPU / CPU，三者均走 **bfloat16** autocast——**不要改回 fp16**，fp16 在 CUDA 上需要 `GradScaler`，否则梯度会静默下溢归零。
- **XPU 加速**：需要 XPU 构建的 torch（`pip install --index-url https://download.pytorch.org/whl/xpu torch`）。从 PyPI 装的默认是 `+cpu` 版本，`torch.xpu.is_available()` 恒为 False。本机 Arc 130T 跑完 6000 步 493 s，CPU（7 线程）910 s。
- token 流常驻目标设备、batch 在设备内 gather；不要改回每步在 host 建 list 再拷贝，那会让 GPU 卡在传输上。
- IPEX 仅用于 CPU 路径。PyTorch 2.5 起 Intel GPU 支持已并入上游，XPU 上不需要 IPEX。
- 训练结束会同时存 `checkpoints/poem_model.pt`（可续训/重导出）与 `data/poem_model.bin`（固件用）。
- **`.pt` checkpoint 绝不能存进 `data/`**：`main/CMakeLists.txt` 的 `spiffs_create_partition_image` 会把整个 `data/` 目录原样打包进 4MB SPIFFS 分区，混进一个 ~2.9MB 的训练 checkpoint 会直接把镜像撑爆，`idf.py build` 报 `SpiffsFullError`。

## 系统架构

- **main/display_driver.h / .c**：屏幕与硬件兼容层。三种板型经 `idf.py menuconfig` 选择：
  - **Waveshare ESP32-S3-Touch-LCD-2**（默认）：板载 2.0 寸 240x320 ST7789 SPI LCD，背光 GPIO 1，40MHz SPI。诗句以 16x16 点阵逐字渲染，每行满 14 字换行，最多 5 行。
  - **Generic SSD1306**：外接 128x64 I2C OLED 单色屏。
  - **Headless**：无屏，仅串口输出。
- **main/zh_font_16x16.c / .h**：自动生成的中文点阵字库，**请勿手工编辑**。
- **main/llama.h**：仅含启动时绘制的 XBM 位图（`llama_bmp`）。
- **main/llm.h**：Transformer 结构体与函数声明：`Config`、`TransformerWeights`、`RunState`、`Transformer`、`Tokenizer`、`Sampler`。全局使用 `v4sf`（16 字节对齐 `float`）适配 ESP-DSP 点积。
- **main/llm.c**：llama2.c `run.c` 的移植：权重加载（malloc 后整文件读入 RAM/PSRAM）、分词器（BPE + UTF-8 字节回退）、前向传播、采样器（贪婪 / 温度 / top-p）与主生成循环。
- **main/main.c**：`app_main` 入口：初始化屏幕（`display_driver_init`）、挂载 SPIFFS（`init_storage`）、从 `/data` 构建 Transformer / Tokenizer / Sampler，轮播提示词调用 `generate(...)` 并实时更新速率。

### 与上游 llama2.c 的关键差异

`encode()` **移除了 SentencePiece 的 dummy 前导空格**。上游会在 BOS 后插一个空格 token，但本模型的训练样本从不以空格开头（空格只作为 `主题：X 体裁：Y` 的字段分隔符出现在中间），保留它会让每次生成都从训练中未见过的上下文起步。改动 `encode()` 时切勿把它加回来。

由于平台不支持真正的 mmap，`read_checkpoint()` 将整个模型文件读入单一 malloc 缓冲区，`memory_map_weights()` 在该缓冲区内移动指针分配权重。`munmap` / `close` 封装为 `free` 或空操作（`custom_munmap` / `custom_close`），仅为保持与上游函数签名兼容。

### 硬件与性能优化

1. **8MB Octal (OPI) PSRAM**：`CONFIG_SPIRAM_MODE_OCT=y`，80MHz，满足权重与 KV Cache 需求。
2. **前向传播双核并发**：`matmul()` 与 `forward()` 内的注意力循环对半拆分，Core 0 / Core 1 绑核 FreeRTOS 任务（`matmul_task` / `forward_task`）并发执行，通过信号量与 `xEventGroupSync` 屏障同步。
3. **ESP-DSP SIMD 点积**：`matmul()` 调用 `dsps_dotprod_f32_aes3`，发挥 ESP32-S3 SIMD 指令优势。
4. **`sdkconfig` 时钟与缓存调优**：CPU 240MHz（`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=240`）、PSRAM 80MHz（`CONFIG_SPIRAM_SPEED_80M`）、指令缓存 32KB（`CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE=0x8000`）。
5. **`linker.lf` 内存布局**：`libmain.a` 整体放入 IRAM/DRAM 提速，但 64KB 的 `zh_font_16x16` 点阵表显式映射回 Flash（`(default)`），避免挤占模型与 KV Cache 所需的 DRAM。

> 实测约 **17.6–18.5 tok/s**（Waveshare ESP32-S3-Touch-LCD-2，词表 2048）。词表 1024 时约 36.6 tok/s，扩表后输出投影层参数翻倍，速率大致减半。
