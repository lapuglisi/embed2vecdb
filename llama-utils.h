#ifndef __EMBED2VECDB_APP_LLAMA_UTILS_H__
#define __EMBED2VECDB_APP_LLAMA_UTILS_H__
#include "app-llama.h"
#include "llama-cpp.h"
#include <string>
#include <vector>

bool app_llama_tokenize(std::vector<llama_token> &, const struct llama_vocab *,
                        const std::string &, bool, bool);

void app_llama_batch_add_seq(llama_batch &, const std::vector<int32_t> &,
                             llama_seq_id);

void app_llama_batch_add(struct llama_batch &, llama_token, llama_pos,
                         const std::vector<llama_seq_id> &, bool);

bool app_llama_batch_decode(llama_context *, llama_batch &, float *, int, int,
                            int);

inline void app_llama_batch_clear(llama_batch &batch)
{
  batch.n_tokens = 0;
}

void app_llama_embd_normalize(const float *, float *, int, int);

std::string app_llama_token_to_piece(const struct llama_vocab *, llama_token,
                                     bool);

#endif // __EMBED2VECDB_APP_LLAMA_UTILS_H__
