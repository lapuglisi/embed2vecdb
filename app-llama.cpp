#include "app-llama.h"
#include "llama.h"
#include "utils.h"
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
	args->qdrant_uri.assign(QdrandDefaultUri);

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
    LOG_ERR("could not load the llama model.\n");
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
  data->embd_sep = "\n";
  data->cls_sep = "\t";
  data->embed_norm = embedding_normalize_algorithm_t::Euclidean;

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

int app_llm_tokenize(const app_llama_data_t &, const std::string &,
                     llama_input_vector_t &)
{
  return 0;
}

bool app_llm_get_embeddings(const app_llama_data_t &, const int,
                            const llama_input_vector_t &, std::vector<float> &)
{
  return true;
}
