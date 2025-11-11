#include "app-llama.h"
#include "utils.h"
#include <stdio.h>

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

  return 0;
}
