#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes the display hardware according to the active board configuration.
 * 
 * @return true if initialized successfully, false otherwise.
 */
bool display_driver_init(void);

/**
 * @brief Displays a status or text message on the screen.
 * 
 * @param text The text to output.
 */
void display_driver_write_text(const char *text);

/**
 * @brief Draws the project logo / Llama graphic on the screen.
 */
void display_driver_draw_logo(void);

/**
 * @brief Displays token generation statistics (tok/s) on the screen.
 * 
 * @param tk_s Tokens per second.
 */
void display_driver_show_stats(float tk_s);

/**
 * @brief Clears the screen.
 */
void display_driver_clear(void);

/**
 * @brief Prepares the screen UI for Chinese poetry generation.
 */
void display_driver_start_poem(void);

/**
 * @brief Appends a single Chinese character token to the on-screen poem layout.
 * 
 * @param token_id The token ID in vocabulary.
 */
void display_driver_append_token(int token_id);

/**
 * @brief Displays the final inference statistics card below the poem.
 * 
 * @param tk_s Tokens per second.
 */
void display_driver_show_poem_complete(float tk_s);

#ifdef __cplusplus
}
#endif
