#include "app-llama.h"
#include "llama-utils.h"
#include "llama.h"
#include "qdrant.h"
#include <cstdint>
#include <math.h>
#include <string.h>
#include <sys/sysinfo.h>

#define APPARGS_PARSE(_idx, _max, _array, _test, _target) \
  if (strcmp(_array[_idx], _test) == 0)                   \
  {                                                       \
    if (++_idx < _max)                                    \
    {                                                     \
      _target(_array[_idx]);                              \
    }                                                     \
  }

// clang-format off
bool app_parse_args(int argc, char **argv, app_llama_args_t *args)
{
	if (NULL == args)
	{
		LOG_ERR("argument 'args' is NULL\n");
		return false;
	}

	// Set defaults
	args->batch_size = 2048;
	args->ubatch_size = 512;
	args->ctx_size = 0; // Use model's
	args->threads = 0;
	args->verbose = false;
	args->n_gpu_layers = 0;
	args->qdrant_uri.assign(QDRANT_DEFAULT_URI);

  for (int i = 1; i < argc; i++)
  {
    APPARGS_PARSE(i, argc, argv, "--model", args->model.assign)
		else APPARGS_PARSE(i, argc, argv, "--source", args->source.assign)
		else APPARGS_PARSE(i, argc, argv, "--ctx", args->ctx_size = std::stoi)
		else APPARGS_PARSE(i, argc, argv, "--ngl", args->n_gpu_layers = std::stoi)
		else APPARGS_PARSE(i, argc, argv, "--n_batch", args->batch_size = std::stoi)
		else APPARGS_PARSE(i, argc, argv, "--n_ubatch", args->ubatch_size = std::stoi)
		else APPARGS_PARSE(i, argc, argv, "--threads", args->threads = std::stoi)
		else APPARGS_PARSE(i, argc, argv, "--qdrant", args->qdrant_uri.assign)
		else if (strcmp(argv[i], "--verbose") == 0)
		{
			args->verbose = true;
		}
  }
	if (args->threads == 0)
	{
		args->threads = ceil(get_nprocs() / 2);
		LOG("params 'threads' not defined, using %d (nproc / 2)\n", args->threads);
	}

	if (args->model.length() == 0)
	{
		LOG_ERR("param --model [MODEL_PATH] is mandatory.\n");
		return false;
	}

  return true;
}
// clang-format on

bool app_llm_init(app_llama_args_t &args, app_llama_data_t *data)
{
  if (NULL == data)
  {
    LOG_ERR("argument 'data' is NULL.\n");
    return false;
  }

  //

  // if the number of prompts that would be encoded is known in advance, it's
  // more efficient to specify the
  //   --parallel argument accordingly. for convenience, if not specified, we
  //   fallback to unified KV cache in order to support any number of prompts

  /*
  if (params.n_parallel == 1)
  {
    LOG_INF("%s: n_parallel == 1 -> unified KV cache is enabled\n", __func__);
    params.kv_unified = true;
  }
  */

  // utilize the full context
  if (args.batch_size < args.ctx_size)
  {
    LOG("info: setting batch size to %d\n", args.ctx_size);
    args.batch_size = args.ctx_size;
  }

  // for non-causal models, batch size must be equal to ubatch size
  /*
  if (params.attention_type != LLAMA_ATTENTION_TYPE_CAUSAL)
  {
    params.n_ubatch = params.n_batch;
  }
  */

  // get max number of sequences per batch
  data->n_seq_max = llama_max_parallel_sequences();

  llama_backend_init();
  // llama_numa_init(params.numa);

  // load the model
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = args.n_gpu_layers;
  // mp.use_mlock, mp.use_mmap ;
  data->model = llama_model_load_from_file(args.model.c_str(), mp);
  if (NULL == data->model)
  {
    LOG_ERR("unable to load model.\n");
    return false;
  }

  // load the context
  llama_context_params cp = llama_context_default_params();
  cp.embeddings = true;
  cp.n_batch = args.batch_size;
  cp.n_ubatch = args.ubatch_size;
  cp.n_threads = args.threads;
  cp.n_ctx = args.ctx_size;
  cp.n_seq_max = 1;

  data->ctx = llama_init_from_model(data->model, cp);
  if (NULL == data->ctx)
  {
    LOG_ERR("unable to load llama context.\n");
    return false;
  }

  llama_model *model = data->model;
  llama_context *ctx = data->ctx;
  const llama_vocab *vocab = llama_model_get_vocab(model);

  const int n_ctx_train = llama_model_n_ctx_train(model);
  const int n_ctx = llama_n_ctx(ctx);

  const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

  if (llama_model_has_encoder(model) && llama_model_has_decoder(model))
  {
    LOG_ERR(
        "computing embeddings in encoder-decoder models is not supported\n");
    return false;
  }

  if (n_ctx > n_ctx_train)
  {
    LOG("warning: model was trained on only %d context tokens (%d "
        "specified)\n",
        n_ctx_train, n_ctx);
  }

  // Set extra values to data
  data->n_batch = args.batch_size;
  data->n_ubatch = args.ubatch_size;
  data->n_seq_max = 1; // TODO: add parameter for that?
  data->embd_sep = "\n";
  data->cls_sep = "\t";
  data->embed_norm = embedding_normalize_algorithm_t::Euclidean;
  data->model_n_embed = llama_model_n_embd(data->model);

  return true;
}

bool app_llm_destroy(app_llama_data_t *data)
{
  if (NULL != data)
  {
    if (NULL != data->ctx)
    {
      LOG("freeing llama context @ %p.\n", data->ctx);
      llama_free(data->ctx);
      data->ctx = NULL;
    }

    if (NULL != data->model)
    {
      LOG("freeing llama model @ %p.\n", data->model);
      llama_model_free(data->model);
      data->model = NULL;
    }
  }

  LOG("freeing the llama backend.\n");
  llama_backend_free();

  return true;
}

int app_llm_tokenize(const app_llama_data_t &data, const std::string &text,
                     llama_input_vector_t &inputs)
{
  llama_model *model = data.model;
  enum llama_pooling_type pooling_type = llama_pooling_type(data.ctx);

  const llama_vocab *vocab = llama_model_get_vocab(data.model);

  if (NULL == vocab)
  {
    LOG_ERR("could not load the model's vocab\n");
    return -1;
  }

  // split the prompt into lines
  std::vector<std::string> prompts = split_lines(text, data.embd_sep);

  // max batch size
  const uint64_t n_batch = data.n_batch;

  // get added sep and eos token, if any
  const std::string added_sep_token =
      llama_vocab_get_add_sep(vocab)
          ? llama_vocab_get_text(vocab, llama_vocab_sep(vocab))
          : "";
  const std::string added_eos_token =
      llama_vocab_get_add_eos(vocab)
          ? llama_vocab_get_text(vocab, llama_vocab_eos(vocab))
          : "";
  const char *rerank_prompt = llama_model_chat_template(model, "rerank");

  // tokenize the prompts and trim
  for (const auto &prompt : prompts)
  {
    std::vector<llama_token> inp;

    // split classification pairs and insert expected separator tokens
    if (pooling_type == LLAMA_POOLING_TYPE_RANK &&
        prompt.find(data.cls_sep) != std::string::npos)
    {
      std::vector<std::string> pairs = split_lines(prompt, data.cls_sep);
      if (rerank_prompt != nullptr)
      {
        const std::string query = pairs[0];
        const std::string doc = pairs[1];
        std::string final_prompt = rerank_prompt;
        string_replace_all(final_prompt, "{query}", query);
        string_replace_all(final_prompt, "{document}", doc);

        if (!app_llama_tokenize(inp, vocab, final_prompt, true, true))
        {
          LOG("warning: app_llama_tokenize failed.\n");
        }
      }
      else
      {
        std::string final_prompt;
        for (size_t i = 0; i < pairs.size(); i++)
        {
          final_prompt += pairs[i];
          if (i != pairs.size() - 1)
          {
            if (!added_eos_token.empty())
            {
              final_prompt += added_eos_token;
            }
            if (!added_sep_token.empty())
            {
              final_prompt += added_sep_token;
            }
          }
        }

        app_llama_tokenize(inp, vocab, final_prompt, true, true);
      }
    }
    else
    {
      app_llama_tokenize(inp, vocab, prompt, true, true);
    }

    if (inp.size() > n_batch)
    {
      LOG_ERR("number of tokens in input line (%lld) exceeds batch size "
              "(%lld), increase batch size and re-run\n",
              (long long int)inp.size(), (long long int)n_batch);
      return -1;
    }

    inputs.push_back(inp);
  }

  // check if the last token is SEP/EOS
  // it should be automatically added by the tokenizer when
  // 'tokenizer.ggml.add_eos_token' is set to 'true'
  for (auto &inp : inputs)
  {
    if (inp.empty() || (inp.back() != llama_vocab_sep(vocab) &&
                        inp.back() != llama_vocab_eos(vocab)))
    {
      LOG("last token in the prompt is not SEP or EOS\n");
      LOG("'tokenizer.ggml.add_eos_token' should be set to 'true' in "
          "the GGUF header\n");
    }
  }

  // tokenization stats
  /*
  if (params.verbose_prompt)
  {
    for (int i = 0; i < (int)inputs.size(); i++)
    {
      LOG_INF("%s: prompt %d: '%s'\n", __func__, i, prompts[i].c_str());
      LOG_INF("%s: number of tokens in prompt = %zu\n", __func__,
              inputs[i].size());
      for (int j = 0; j < (int)inputs[i].size(); j++)
      {
        LOG("%6d -> '%s'\n", inputs[i][j],
            common_token_to_piece(ctx, inputs[i][j]).c_str());
      }
      LOG("\n\n");
    }
  }
  */

  return prompts.size();
}

bool app_llm_get_embeddings(const app_llama_data_t &data, const int n_prompts,
                            const llama_input_vector_t &inputs,
                            std::vector<float> &embeddings)
{
  // initialize batch
  enum llama_pooling_type pooling_type = llama_pooling_type(data.ctx);
  const int32_t n_batch = data.n_batch;
  const int32_t n_seq_max = data.n_seq_max;

  struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

  // count number of embeddings
  int n_embd_count = 0;
  if (pooling_type == LLAMA_POOLING_TYPE_NONE)
  {
    for (int k = 0; k < n_prompts; k++)
    {
      n_embd_count += inputs[k].size();
    }
  }
  else
  {
    n_embd_count = n_prompts;
  }

  // allocate output
  const int n_embd = llama_model_n_embd(data.model);

  const size_t embd_size = n_embd_count * n_embd;
  embeddings.resize(embd_size);
  embeddings.assign(embd_size, 0.0);

  float *emb = embeddings.data();

  // break into batches
  int e = 0; // number of embeddings already stored
  int s = 0; // number of prompts in current batch
  for (int k = 0; k < n_prompts; k++)
  {
    // clamp to n_batch tokens
    auto &inp = inputs[k];

    const uint64_t n_toks = inp.size();

    // encode if at capacity
    if (batch.n_tokens + n_toks > n_batch || s >= n_seq_max)
    {
      float *out = emb + e * n_embd;
      app_llama_batch_decode(data.ctx, batch, out, s, n_embd, data.embed_norm);

      e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
      s = 0;

      app_llama_batch_clear(batch);
    }

    // add to batch
    // TODO: app_llama_batch_add_seq(batch, inp, s);
    app_llama_batch_add_seq(batch, inp, s);
    s += 1;
  }

  // final batch
  float *out = emb + e * n_embd;
  app_llama_batch_decode(data.ctx, batch, out, s, n_embd, data.embed_norm);

  return true;
}
