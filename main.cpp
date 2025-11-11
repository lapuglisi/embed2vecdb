#include "app-llama.h"
#include "qdrant.h"
#include "utils.h"
#include <stdio.h>
#include <uuid/uuid.h>

int main(int argc, char **argv)
{
  printf(":: embed2vecdb ::\n");

  app_llama_args_t args;
  if (!app_parse_args(argc, argv, &args))
  {
    LOG_ERR("could not parse arguments.\n");
    return 1;
  }

  if (args.verbose)
  {
    printf("\n");
    printf("model ......... %s\n", args.model.c_str());
    printf("source ........ %s\n", args.source.c_str());
    printf("qdrant_uri .... %s\n", args.qdrant_uri.c_str());
    printf("ctx_size ...... %d\n", args.ctx_size);
    printf("batch_size .... %d\n", args.batch_size);
    printf("ubatch_size ... %d\n", args.ubatch_size);
    printf("threads ....... %d\n", args.threads);
    printf("n_gpu_layers .. %d\n", args.n_gpu_layers);
    printf("\n");
  }

  // Init app_llm
  app_llama_data_t data;
  if (!app_llm_init(args, &data))
  {
    LOG_ERR("could not initialize.\n");
    app_llm_destroy(&data);

    return -1;
  }

  // Test for qdrant connection
  qdrant_info_t info;
  if (!qdrant_init(args.qdrant_uri, &info))
  {
    LOG_ERR("qdrant_init failed.\n");
    return -1;
  }

  qdrant_point_array_t points;
  points.push_back({.id = generate_uuid(),
                    .payload_x = "key",
                    .payload_y = "value",
                    .vector = {1, 1, 1, 1}});

  qdrant_points_insert(info, points);
  return 0;

  llama_input_vector_t result;
  std::string text(
      "serominers seroclevers serowonders\nseroflatos\nserocomidas");

  LOG("getting embeddings for '%s'.\n", text.c_str());

  int n_prompts = app_llm_tokenize(data, text, result);
  if (n_prompts <= 0)
  {
    LOG_ERR("could not tokenize the input string '%s'\n", text.c_str());
  }
  else
  {
    std::vector<float> embeddings;
    if (!app_llm_get_embeddings(data, n_prompts, result, embeddings))
    {
      LOG_ERR("could not get embeddings.\n");
    }
    else
    {
      printf("\n\n");
      for (int s = 0; s < embeddings.size(); s++)
      {
        printf("%.5f, ", embeddings.at(s));
        if (s % 10 == 0)
        {
          printf("\n");
        }
      }
      printf("\n\n");
    }
  }

  qdrant_destroy(&info);
  app_llm_destroy(&data);

  return 0;
}
