# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在处理本仓库代码时提供指引与参考。

## 项目概述

本项目是 [llama2.c](https://github.com/karpathy/llama2.c) 的 ESP32-S3 固件移植版本。它在微控制器上直接运行一个超微型（260K 参数）Llama-2 架构 Transformer 模型，逐 token 生成文本，并在连接的 SSD1306 OLED 屏幕上实时显示每秒生成的 token 数量（tokens/sec）。所用模型为 [tinyllamas stories260K checkpoint](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K)。

目标硬件为 ESP32-S3FH4R2（需要内置 PSRAM，模型与 KV 缓存共需约 1MB 内存）。显示屏为通过 I2C 连接的 128x64 SSD1306 OLED（默认引脚为 SDA=GPIO8, SCL=GPIO9，可在 `idf.py menuconfig` 中的 "Example Configuration > I2C Master" 进行配置）。

## 编译与烧录命令

依赖 [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html#installation) 工具链（目标芯片为 `esp32s3`，IDF 组件管理器锁定版本为 `5.3.0`，详情见 `dependencies.lock`）。

在 Windows PowerShell 环境下，执行 `idf.py` 相关命令前需先加载 ESP-IDF 环境变量：

```powershell
. C:\Espressif\tools\Microsoft.v6.0.1.PowerShell_profile.ps1
```

常用命令：

```bash
idf.py build
idf.py -p COMx flash
idf.py -p COMx monitor
```

（在 Linux/macOS 下请将 `COMx` 替换为对应设备路径，如 `/dev/ttyUSB0`）

本项目不包含独立的单元测试套件或代码检查（lint）配置。作为嵌入式固件项目，代码正确性通过编译并烧录至真实硬件验证（或通过串口监视器查看文本生成输出及上报的 tok/s 速度）。

`data/` 目录（包含 `stories260K.bin` 权重文件和 `tok512.bin` 分词器文件）会被打包为 SPIFFS 镜像并烧录至独立分区（参考 `main/CMakeLists.txt` 中的 `spiffs_create_partition_image` 调用与 `partitions.csv`）。若修改了 `data/` 中的内容，只要镜像重新构建（属于常规构建流程的一部分），执行 `idf.py flash` 也会同步重新烧录 SPIFFS 分区。

## 系统架构

- **main/display_driver.h / main/display_driver.c**：屏幕与硬件兼容层抽象。支持通过 `idf.py menuconfig` 选择目标板型：
  - **Waveshare ESP32-S3-Touch-LCD-2**：板载 2.0寸 240x320 ST7789 SPI LCD，支持背光控制（GPIO 1）与 40MHz 高速 SPI。
  - **Generic SSD1306**：外接 128x64 I2C OLED 单色屏。
  - **Headless**：无屏幕模式，仅串口输出。
- **main/llama.h**：仅包含启动时绘制到屏幕的原始 XBM 位图（`llama_bmp`）。
- **main/llm.h**：Transformer 相关的结构体、类型定义与函数声明：`Config`、`TransformerWeights`、`RunState`、`Transformer`、`Tokenizer`、`Sampler`。全局使用 `v4sf` 类型（16 字节对齐的 `float`）以适配 ESP-DSP 点积运算。
- **main/llm.c**：llama2.c 的 `run.c` 移植实现：模型权重加载（通过 malloc 分配内存并将整个文件读取到 RAM/PSRAM）、分词器（BPE，支持 UTF-8 字节回退）、前向传播、采样器（贪婪采样 / 温度采样 / top-p 采样）以及主生成循环。
- **main/main.c**：`app_main` 入口：初始化屏幕抽象层（`display_driver_init`）、挂载 SPIFFS（`init_storage`）、从 `/data` 读取文件构建 Transformer、Tokenizer 与 Sampler、在屏幕上显示状态与羊驼图腾，随后调用 `generate(...)` 进行推理并实时更新生成速率。

### 硬件与性能优化

1. **8MB Octal (OPI) PSRAM 支持**：配置启用 80MHz 八线制 PSRAM（`CONFIG_SPIRAM_MODE_OCT=y`），满足 Transformer 权重加载和 KV Cache 内存需求。
2. **前向传播双核并发计算**：`matmul()` 和 `forward()` 内的注意力循环均将计算任务对半拆分：Core 0 与 Core 1 绑定的 FreeRTOS 任务（`matmul_task` / `forward_task`）并发执行，通过信号量与 `xEventGroupSync` 屏障同步。
3. **ESP-DSP SIMD 点积加速**：`matmul()` 调用 `dsps_dotprod_f32_aes3`（ESP-DSP 组件），充分发挥 ESP32-S3 的 SIMD 指令集优势，实测生成速率可达 **~36.6 tok/s**。
4. **`sdkconfig` 时钟与缓存调优**：CPU 频率设为 240MHz（`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=240`），PSRAM 设为 80MHz（`CONFIG_SPIRAM_SPEED_80M`），放大了指令缓存（`CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE=0x8000`）。`linker.lf` 将 `main` 归档放置在 IRAM/DRAM 中提速。

由于平台不支持真正的文件 mmap，`read_checkpoint()` 会将整个模型文件读取到单一 malloc 缓冲区中，`memory_map_weights()` 则直接在该缓冲区中移动指针分配权重。`munmap` 和 `close` 则封装为 `free` 或空操作（`custom_munmap`/`custom_close`），仅用于保持与上游 llama2.c 的函数签名兼容。

## 修改文本生成参数

采样参数（`temperature`、`topp`、`steps`、`prompt`、`rng_seed`）直接硬编码在 `main/main.c` 的 `app_main()` 中，而非通过命令行参数解析（微控制器环境无命令行接口）。若需调整生成行为，请直接修改该处的变量值。
