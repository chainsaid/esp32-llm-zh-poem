#include <stdio.h>
#include <inttypes.h>
#include <time.h>
#include <string.h>
#include "esp_spiffs.h"
#include "sdkconfig.h"
#include "esp_err.h"
#include "esp_log.h"
#include "llm.h"
#include "display_driver.h"
#include "imu_driver.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"

static const char *TAG = "MAIN";

// BOOT button (GPIO0 on every ESP32-S3 dev board, active-low) toggles the
// meter live while the poem chain keeps running -- no reflash needed.
// GPIO0 is safe to repurpose here: the Waveshare LCD driver hardcodes
// reset_gpio_num = -1 and never touches this pin (see display_driver.c).
#define METER_TOGGLE_GPIO GPIO_NUM_0
#define METER_5YAN 5
#define METER_7YAN 7

static volatile int g_meter_chars = METER_5YAN; // 五言/七言, read fresh at the start of every line
static volatile TickType_t g_last_toggle_tick = 0;

static void toggle_meter(void)
{
    g_meter_chars = (g_meter_chars == METER_5YAN) ? METER_7YAN : METER_5YAN;
}

static void IRAM_ATTR meter_toggle_isr(void *arg)
{
    // Mechanical button bounce is a few ms; debounce with a hard cooldown.
    TickType_t now = xTaskGetTickCountFromISR();
    if (now - g_last_toggle_tick < pdMS_TO_TICKS(300))
    {
        return;
    }
    g_last_toggle_tick = now;
    toggle_meter();
}

static void meter_toggle_button_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = 1ULL << METER_TOGGLE_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE,
    };
    gpio_config(&io_conf);
    gpio_install_isr_service(0);
    gpio_isr_handler_add(METER_TOGGLE_GPIO, meter_toggle_isr, NULL);
}

/**
 * @brief Initializes SPIFFS storage
 */
void init_storage(void)
{
    ESP_LOGI(TAG, "Initializing SPIFFS...");

    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/data",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = false
    };

    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK)
    {
        if (ret == ESP_FAIL)
        {
            ESP_LOGE(TAG, "Failed to mount or format filesystem");
        }
        else if (ret == ESP_ERR_NOT_FOUND)
        {
            ESP_LOGE(TAG, "Failed to find SPIFFS partition");
        }
        else
        {
            ESP_LOGE(TAG, "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
        }
        return;
    }

    size_t total = 0, used = 0;
    ret = esp_spiffs_info(NULL, &total, &used);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to get SPIFFS partition information (%s)", esp_err_to_name(ret));
    }
    else
    {
        ESP_LOGI(TAG, "Partition size: total: %d, used: %d", total, used);
    }
}

/**
 * @brief Callback once token generation is complete
 * 
 * @param tk_s The number of tokens per second generated
 */
void generate_complete_cb(float tk_s)
{
    // Serial only -- the screen stays on the poem stream, no stats card overlay.
    ESP_LOGI(TAG, "Chinese Poetry Inference speed: %.2f tok/s", tk_s);
}

// Renders "体裁：五言/七言，主题：X" in the status strip below the poem
// card. Called only when meter or theme actually changed (see app_main's
// shown_meter/shown_theme_idx tracking) since it mallocs via encode().
static void update_status_display(Tokenizer *tokenizer, int meter_chars, const char *theme)
{
    const char *meter_label = (meter_chars == METER_7YAN) ? "七言" : "五言";
    char status_buf[64];
    snprintf(status_buf, sizeof(status_buf), "体裁：%s，主题：%s", meter_label, theme);

    int tokens[32];
    int n_tokens = 0;
    encode(tokenizer, status_buf, 0, 0, tokens, &n_tokens);
    display_driver_show_status(tokens, n_tokens);
}

void app_main(void)
{
    ESP_LOGI(TAG, "Starting ESP32-S3 Chinese Poetry LLM App...");

    // Initialize display hardware abstraction layer
    display_driver_init();
    display_driver_write_text("Loading Poem LLM...");

    // Mount SPIFFS filesystem
    init_storage();

    // Chinese Poetry Model parameters
    char *checkpoint_path = "/data/poem_model.bin";
    char *tokenizer_path = "/data/poem_tok.bin";
    float temperature = 0.8f;        // 0.8 = optimal creativity for poetry
    float topp = 0.9f;               // top-p nucleus sampling
    unsigned long long rng_seed = 0; // seed rng with time

    if (rng_seed <= 0)
        rng_seed = (unsigned int)time(NULL);

    // Build the Transformer via the model .bin file
    Transformer transformer;
    ESP_LOGI(TAG, "Loading Chinese Poetry LLM checkpoint from %s", checkpoint_path);
    build_transformer(&transformer, checkpoint_path);

    // Build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // Build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // BOOT button toggles 五言/七言 live; g_meter_chars is read fresh at the
    // start of every line below, so a press mid-run takes effect on the very
    // next line without needing a reflash. The onboard QMI8658 IMU, if
    // found, adds the same control via shake gestures (up-down -> meter,
    // left-right -> re-seed theme) -- see imu_driver.c. imu_driver_init()
    // returning false (sensor not found / wrong pins) is non-fatal: the
    // stream just runs without gesture control, BOOT button still works.
    meter_toggle_button_init();
    imu_driver_init();

    // Themes to seed the first line, and to re-seed on a left-right shake.
    // Outside of those moments, every line is conditioned solely on the
    // previous line's text (体裁 tag + 上一句), matching the "仅体裁"
    // unconditioned-continuation training samples in tools/dataset.py --
    // there is no running 主题 carried forward, so the subject naturally
    // drifts down the chain like a renga rather than staying pinned to one
    // topic forever. That's intentional: the point of this mode is an
    // unbounded stream, not a fixed-length themed poem.
    static const char *seed_themes[] = {
        "明月", "春风", "边塞", "相思", "孤舟", "江南", "山水", "夕阳", "故乡", "落花"
    };
    int num_seed_themes = sizeof(seed_themes) / sizeof(seed_themes[0]);
    int seed_idx = (int)(esp_random() % num_seed_themes);
    bool reseed_theme = true; // consumed on line 1, then only set again by a shake gesture

    char prev_line[128] = {0};
    char prompt_buf[192];
    int line_idx = 1; // 1-based; odd line forces '，', even line forces '。' (couplet alternation)
    int tokens_since_report = 0;
    TickType_t report_start_tick = xTaskGetTickCount();
    int shown_meter = -1;     // sentinel so the status bar always draws on the first line
    int shown_theme_idx = -1;

    display_driver_start_poem();
    printf("\n================ [ 无限诗词流 ] ================\n");

    while (1)
    {
        switch (imu_driver_poll_gesture())
        {
        case IMU_GESTURE_UPDOWN:
            toggle_meter();
            ESP_LOGI(TAG, "IMU shake (up-down): meter toggled");
            break;
        case IMU_GESTURE_LEFTRIGHT:
        {
            int new_idx = (int)(esp_random() % num_seed_themes);
            if (new_idx == seed_idx) new_idx = (new_idx + 1) % num_seed_themes;
            seed_idx = new_idx;
            reseed_theme = true;
            ESP_LOGI(TAG, "IMU shake (left-right): theme re-seeded to %s", seed_themes[seed_idx]);
            break;
        }
        default:
            break;
        }

        int meter_chars = g_meter_chars; // snapshot: may be toggled by the BOOT button/IMU between lines
        const char *meter_label = (meter_chars == METER_7YAN) ? "七言" : "五言";

        if (meter_chars != shown_meter || seed_idx != shown_theme_idx)
        {
            update_status_display(&tokenizer, meter_chars, seed_themes[seed_idx]);
            shown_meter = meter_chars;
            shown_theme_idx = seed_idx;
        }

        if (reseed_theme)
        {
            snprintf(prompt_buf, sizeof(prompt_buf), "主题：%s 体裁：%s\n", seed_themes[seed_idx], meter_label);
            reseed_theme = false;
        }
        else
        {
            snprintf(prompt_buf, sizeof(prompt_buf), "体裁：%s\n%s", meter_label, prev_line);
        }

        // Fresh randomness for every line, same as the old per-poem reseed.
        sampler.rng_state = (unsigned int)time(NULL) + (unsigned int)esp_random();

        int is_couplet_end = (line_idx % 2 == 0);
        int n = generate_poem_line(&transformer, &tokenizer, &sampler, prompt_buf,
                                    meter_chars, is_couplet_end, prev_line, sizeof(prev_line));
        if (n <= 0)
        {
            ESP_LOGE(TAG, "generate_poem_line produced no output, retrying line %d", line_idx);
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        // No forced newline here: generate_poem_line() already streamed this
        // line's characters (comma/period included) straight onto the
        // display as they were generated. Text just keeps flowing left to
        // right and only wraps to the next row when a row runs out of
        // width (see display_driver_append_token()'s auto-wrap), scrolling
        // once the card is full -- no per-line breaks.
        tokens_since_report += n;
        line_idx++;

        // Serial-only tok/s reporting, decoupled from the display now that
        // it scrolls instead of paging.
        if (line_idx % 7 == 1)
        {
            TickType_t now = xTaskGetTickCount();
            uint32_t elapsed_ms = (now - report_start_tick) * portTICK_PERIOD_MS;
            if (elapsed_ms > 0)
            {
                float tks = tokens_since_report / (elapsed_ms / 1000.0f);
                generate_complete_cb(tks);
            }
            tokens_since_report = 0;
            report_start_tick = xTaskGetTickCount();
        }
    }
}
