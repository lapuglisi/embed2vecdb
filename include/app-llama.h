
#ifndef _EMBED2VECDB_APP_LLAMA_H_
#define _EMBED2VECDB_APP_LLAMA_H_

#include "llama.h"
#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>

typedef enum _embed_norm_algo
{
  NoNormalisation = -1,
  MaxAbsolute = 0,
  Euclidean = 2
} embedding_normalize_algorithm_t;

typedef struct _app_llama_args
{
  std::string model;
  std::string source;
  int32_t ctx_size;
  int32_t n_gpu_layers;
  std::string qdrant_uri;
  ushort batch_size;
  ushort ubatch_size;
  ushort threads;
  bool verbose;
} app_llama_args_t;

typedef struct _app_llama_data
{
  llama_model *model;
  llama_context *ctx;
  std::string cls_sep;
  std::string embd_sep;
  int32_t n_batch;
  int32_t n_ubatch;
  int32_t n_seq_max;
  int32_t embed_norm;
  int32_t model_n_embed;
} app_llama_data_t;

typedef std::vector<std::vector<int32_t>> llama_input_vector_t;

bool app_parse_args(int, char **, app_llama_args_t *);

bool app_llm_init(app_llama_args_t &, app_llama_data_t *);

bool app_llm_destroy(app_llama_data_t *);

int app_llm_tokenize(const app_llama_data_t &, const std::string &,
                     llama_input_vector_t &);

bool app_llm_get_embeddings(const app_llama_data_t &, const int,
                            const llama_input_vector_t &, std::vector<float> &);

#endif // _EMBED2VECDB_APP_LLAMA_H_
