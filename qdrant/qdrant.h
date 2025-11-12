#ifndef __EMBED2VECDB_QDRANT_H__
#define __EMBED2VECDB_QDRANT_H__

#include "curl/curl.h"
#include "nlohmann/json.hpp"
#include "utils.h"
#include <string>
#include <vector>

#define QDRANT_DEFAULT_URI "http://localhost:6333"

// qdrant API paths
#define QDRANT_COLLECTIONS_PATH "/collections/{collection_name}"
/* {"vectors": {"size": 4, "distance": "Cosine"}}
 * PUT: Creates collection
 * GET: retrieves collection stats
 */

#define QDRANT_POINTS_SEARCH_PATH "/collections/{collection_name}/points/search"
/* POST: Search for points
 * {"vector": [0.2, 0.5, 0.2, 0.8], "limit": 2}
 */

#define QDRANT_POINTS_INSERT_PATH "/collections/{collection_name}/points"
/* PUT: Insert points
 * {"points": [{"id": 2, "payload": {"caga": "muito"}, "vector": [0.32, 0.75,
 * 0.32, 0.91]}]}
 *
 * POST: Retrieve points stats
 * {ids: [a, b, c]}
 */

typedef enum _qdrant_distance_type
{
  DotProduct = 1,
  Cosine,
  Euclid,
  Manhattan
} qdrant_distance_type_t;

std::string qdrant_get_distance(qdrant_distance_type_t);

typedef struct _qdrant_info
{
  std::string URI;
} qdrant_info_t;

typedef struct _qdrant_collection_info
{
  std::string name;
  unsigned int size;
  qdrant_distance_type_t distance;
} qdrant_colection_info_t;

typedef struct _qdrant_point_spec
{
  std::string id;
  std::string payload_x;
  std::string payload_y;
  std::vector<float> vector;
} qdrant_point_spec_t;

#define QDRANT_WRITE_DATA_SIZE 1024

typedef struct _curl_write_data
{
  void *pointer;
  size_t pointer_len;
} curl_write_data_t;

inline curl_write_data_t *qdrant_malloc_write_data(size_t dsize = 512)
{
  curl_write_data_t *data = (curl_write_data_t *)malloc(
      QDRANT_WRITE_DATA_SIZE * sizeof(curl_write_data_t));

  if (data != NULL)
  {
    data->pointer_len = dsize;
    data->pointer = malloc(data->pointer_len);
  }

  return data;
}

inline void free_write_data(curl_write_data_t *data)
{
  if (data != NULL)
  {
    if (data->pointer != NULL)
    {
      LOG("freeing [curl_write_data_t*]->pointer.\n");
      free(data->pointer);
      data->pointer = NULL;
    }

    LOG("freeing [curl_write_data_t*].\n");
    free(data);
  }
}

typedef std::vector<qdrant_point_spec_t> qdrant_point_array_t;

int qdrant_curl_callback_nop(char *, size_t, size_t, void *);
int qdrant_curl_write_data(char *, size_t, size_t, void *);

bool qdrant_init(const std::string &, qdrant_info_t *);

bool qdrant_collection_create(const qdrant_info_t &,
                              const qdrant_colection_info_t &);
bool qdrant_collection_delete(const qdrant_info_t &,
                              const qdrant_colection_info_t &);

/* Points implementation */
bool qdrant_points_insert(const qdrant_info_t &info,
                          const qdrant_colection_info_t &col,
                          const qdrant_point_array_t &points);

#endif // __EMBED2VECDB_QDRANT_H__
