#ifndef LLM_H
#define LLM_H

/**
 * Original author of this:
 * https://github.com/karpathy/llama2.c 
 * 
 * Slight modifications added to make it ESP32 friendly
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

typedef float v4sf;

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    v4sf* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    v4sf* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    v4sf* rms_att_weight; // (layer, dim) rmsnorm weights
    v4sf* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    v4sf* wq; // (layer, dim, n_heads * head_size)
    v4sf* wk; // (layer, dim, n_kv_heads * head_size)
    v4sf* wv; // (layer, dim, n_kv_heads * head_size)
    v4sf* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    v4sf* w1; // (layer, hidden_dim, dim)
    v4sf* w2; // (layer, dim, hidden_dim)
    v4sf* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    v4sf* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    v4sf* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    v4sf *x; // activation at current time stamp (dim,)
    v4sf *xb; // same, but inside a residual branch (dim,)
    v4sf *xb2; // an additional buffer just for convenience (dim,)
    v4sf *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    v4sf *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    v4sf *q; // query (dim,)
    v4sf *k; // key (dim,)
    v4sf *v; // value (dim,)
    v4sf *att; // buffer for scores/attention values (n_heads, seq_len)
    v4sf *logits; // output logits
    // kv cache
    v4sf* key_cache;   // (layer, seq_len, dim)
    v4sf* value_cache; // (layer, seq_len, dim)
} RunState;


typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    v4sf* data; // memory mapped data pointer
    size_t file_size; // size of the checkpoint file in bytes
} Transformer;



void build_transformer(Transformer *t, char* checkpoint_path);
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);

// Generates exactly one poem line (see llm.c for the full contract): primes
// the KV cache with `prompt`, then samples/forces tokens until `meter_chars`
// content characters are produced, ending the line with '，' (is_couplet_end
// == 0) or '。' (is_couplet_end != 0). Writes the line's raw UTF-8 text (no
// header) into out_buf and returns the content token count, or -1 on a
// prompt encoding failure.
int generate_poem_line(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                        const char *prompt, int meter_chars, int is_couplet_end,
                        char *out_buf, size_t out_buf_cap);

// Encodes `text` (UTF-8) into `tokens` (upper-bounded by strlen(text)+3 --
// caller-allocated). Exposed so main.c can turn a short status string (e.g.
// current 体裁/主题) into token ids for display_driver_show_status(), which
// looks glyphs up by token id the same way the poem card does.
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);

void free_sampler(Sampler* sampler);
void free_transformer(Transformer* t);
void free_tokenizer(Tokenizer* t);


#endif