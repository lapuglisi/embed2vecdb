#include "llama-utils.h"
#include "utils.h"
#include <cmath>
#include <limits>

bool app_llama_tokenize(std::vector<llama_token> &tokens,
                        const struct llama_vocab *vocab,
                        const std::string &text, bool add_special,
                        bool parse_special)
{
  // upper limit for the number of tokens
  int n_tokens = text.length() + 2 * add_special;

  tokens.resize(n_tokens);

  n_tokens = llama_tokenize(vocab, text.data(), text.length(), tokens.data(),
                            tokens.size(), add_special, parse_special);

  if (n_tokens == std::numeric_limits<int32_t>::min())
  {
    LOG_ERR("Tokenization failed: input text too large, "
            "tokenization result exceeds int32_t limit");
    return false;
  }

  if (n_tokens < 0)
  {
    tokens.resize(-n_tokens);
    int check = llama_tokenize(vocab, text.data(), text.length(), tokens.data(),
                               tokens.size(), add_special, parse_special);
    if (check != -n_tokens)
    {
      LOG_ERR("tokenize failed: check != -n_tokens\n");
    }
  }
  else
  {
    tokens.resize(n_tokens);
  }

  return true;
}

void app_llama_batch_add_seq(llama_batch &batch,
                             const std::vector<int32_t> &tokens,
                             llama_seq_id seq_id)
{
  size_t n_tokens = tokens.size();
  for (size_t i = 0; i < n_tokens; i++)
  {
    app_llama_batch_add(batch, tokens[i], i, {seq_id}, true);
  }
}

void app_llama_batch_add(struct llama_batch &batch, llama_token id,
                         llama_pos pos,
                         const std::vector<llama_seq_id> &seq_ids, bool logits)
{
  if (!batch.seq_id[batch.n_tokens])
  {
    LOG_ERR("llama_batch size exceeded\n");
    return;
  }

  batch.token[batch.n_tokens] = id;
  batch.pos[batch.n_tokens] = pos;
  batch.n_seq_id[batch.n_tokens] = seq_ids.size();

  for (size_t i = 0; i < seq_ids.size(); ++i)
  {
    batch.seq_id[batch.n_tokens][i] = seq_ids[i];
  }
  batch.logits[batch.n_tokens] = logits;

  batch.n_tokens++;
}

bool app_llama_batch_decode(llama_context *ctx, llama_batch &batch,
                            float *output, int n_seq, int n_embd, int embd_norm)
{
  const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

  // clear previous kv_cache values (irrelevant for embeddings)
  // llama_memory_clear(llama_get_memory(ctx), true);

  // run model
  LOG("n_tokens = %d, n_seq = %d\n", batch.n_tokens, n_seq);
  if (llama_decode(ctx, batch) < 0)
  {
    LOG_ERR("llama_decode failed to process\n");
  }

  for (int i = 0; i < batch.n_tokens; i++)
  {
    if (!batch.logits[i])
    {
      continue;
    }

    const float *embd = nullptr;
    int embd_pos = 0;

    if (pooling_type == LLAMA_POOLING_TYPE_NONE)
    {
      // try to get token embeddings
      embd = llama_get_embeddings_ith(ctx, i);
      embd_pos = i;

      if (NULL == embd)
      {
        LOG_ERR("failed to get token embeddings\n");
        break;
      }
    }
    else
    {
      // try to get sequence embeddings - supported only when pooling_type is
      // not NONE
      embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
      embd_pos = batch.seq_id[i][0];

      if (NULL == embd)
      {
        LOG_ERR("failed to get token embeddings\n");
        break;
      }
    }

    float *out = output + embd_pos * n_embd;
    app_llama_embd_normalize(embd, out, n_embd, embd_norm);
  }

  return true;
}

void app_llama_embd_normalize(const float *inp, float *out, int n,
                              int embd_norm)
{
  double sum = 0.0;

  switch (embd_norm)
  {
  case -1: // no normalisation
  {
    sum = 1.0;
    break;
  }
  case 0: // max absolute
  {
    for (int i = 0; i < n; i++)
    {
      if (sum < std::abs(inp[i]))
      {
        sum = std::abs(inp[i]);
      }
    }
    sum /= 32760.0; // make an int16 range
    break;
  }
  case 2: // euclidean
  {
    for (int i = 0; i < n; i++)
    {
      sum += inp[i] * inp[i];
    }
    sum = std::sqrt(sum);
    break;
  }
  default: // p-norm (euclidean is p-norm p=2)
  {
    for (int i = 0; i < n; i++)
    {
      sum += std::pow(std::abs(inp[i]), embd_norm);
    }
    sum = std::pow(sum, 1.0 / embd_norm);
    break;
  }
  }

  const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

  for (int i = 0; i < n; i++)
  {
    out[i] = inp[i] * norm;
  }
}

std::string app_llama_token_to_piece(const struct llama_vocab *vocab,
                                     llama_token token, bool special)
{
  std::string piece;
  piece.resize(
      piece.capacity()); // using string internal cache, 15 bytes + '\n'

  const int n_chars =
      llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);

  if (n_chars < 0)
  {
    piece.resize(-n_chars);
    int check =
        llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (check != -n_chars)
    {
      LOG_ERR("token_to_piece failed: check != -n_chars.\n");
    }
  }
  else
  {
    piece.resize(n_chars);
  }

  return piece;
}
