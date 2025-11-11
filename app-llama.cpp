#include "app-llama.h"
#include "llama.h"
#include "utils.h"

bool app_parse_args(int argc, char **argv, app_llama_args_t *args)
{
  return true;
}

bool app_llm_init(app_llama_args_t &args, app_llama_data_t *data)
{
  return true;
}

bool app_llm_destroy(app_llama_data_t *data)
{
  if (NULL != data)
  {
    if (NULL != data->ctx)
    {
      LOG("freeing llama context %p.\n", data->ctx);
      llama_free(data->ctx);
    }
    llama_backend_free();
    return true;
  }

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
