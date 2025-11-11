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

  llama_input_vector_t result;
  std::string text("serominers sao brasileiros");

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
      qdrant_colection_info_t col;
      col.name = "serominers";

      qdrant_point_array_t points;
      qdrant_point_spec_t point;

      point.id = generate_uuid();
      point.payload_x = "sero";
      point.payload_y = "miners";
      point.vector.assign(embeddings.begin(), embeddings.end());

      points.push_back(point);

      qdrant_points_insert(info, col, points);
    }
  }

  app_llm_destroy(&data);

  return 0;
}
