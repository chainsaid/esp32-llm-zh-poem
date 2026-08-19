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
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "MAIN";

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
    ESP_LOGI(TAG, "Chinese Poetry Inference finished! Speed: %.2f tok/s", tk_s);
    display_driver_show_poem_complete(tk_s);
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
    int steps = 110;                 // budget covers prompt + up to a full 七律 (~78 tokens)
    unsigned long long rng_seed = 0; // seed rng with time

    if (rng_seed <= 0)
        rng_seed = (unsigned int)time(NULL);

    // Build the Transformer via the model .bin file
    Transformer transformer;
    ESP_LOGI(TAG, "Loading Chinese Poetry LLM checkpoint from %s", checkpoint_path);
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len;

    // Build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // Build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // Built-in theme+meter conditioning prompts, matching the
    // "主题：{关键词} 体裁：{五绝/七绝/五律/七律/五言/七言}\n" training format
    // (see tools/dataset.py). A bare first-line prompt like the upstream
    // llama2.c demo used is out-of-distribution for this model.
    const char *prompts[] = {
        "主题：明月 体裁：五绝\n",
        "主题：春风 体裁：七绝\n",
        "主题：边塞 体裁：七律\n",
        "主题：相思 体裁：五律\n",
        "主题：孤舟 体裁：五绝\n",
        "主题：江南 体裁：七绝\n",
        "主题：山水 体裁：五言\n",
        "主题：夕阳 体裁：七言\n",
        "体裁：五绝\n",
        "主题：故乡\n",
        NULL // free generation without prompt
    };
    int num_prompts = sizeof(prompts) / sizeof(prompts[0]);
    int poem_counter = 0;

    while (1)
    {
        const char *cur_prompt = prompts[poem_counter % num_prompts];
        poem_counter++;

        // Update random seed for novel poem generation each time
        sampler.rng_state = (unsigned int)time(NULL) + (unsigned int)esp_random();

        // Prepare classical poetry display layout
        display_driver_start_poem();

        // Run inference!
        ESP_LOGI(TAG, ">>> [Poem #%d] Generating Chinese Poem (Prompt: %s) <<<", 
                 poem_counter, cur_prompt ? cur_prompt : "(Free Composition)");
        printf("\n================ [ 诗词创作 第 %d 首 ] ================\n", poem_counter);
        generate(&transformer, &tokenizer, &sampler, (char *)cur_prompt, steps, &generate_complete_cb);
        printf("=========================================================\n\n");

        // Pause for 5 seconds to let the user read before next poem
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
