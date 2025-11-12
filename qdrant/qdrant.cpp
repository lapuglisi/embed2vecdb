#include "qdrant.h"
#include "curl/curl.h"
#include "nlohmann/json.hpp"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <exception>
#include <uuid/uuid.h>

bool qdrant_init(const std::string &qdrant_uri, qdrant_info_t *info)
{
  bool success = true;
  if (NULL == info)
  {
    LOG_ERR("argument 'info' is NULL.\n");
    return false;
  }

  info->URI.assign(qdrant_uri);

  // Call curl_global_init
  CURLcode res = curl_global_init(CURL_GLOBAL_ALL);
  if (res != 0)
  {
    LOG_ERR("curl_global_init failed: %d.\n", res);
    return false;
  }

  CURL *curl = curl_easy_init();
  if (NULL == curl)
  {
    LOG_ERR("curl_easy_init failed.\n");
    curl_global_cleanup();
    return false;
  }

  // Now check if qdrant is online
  curl_easy_setopt(curl, CURLOPT_URL, info->URI.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);

  res = curl_easy_perform(curl);
  success = (CURLE_OK == res);

  if (!success)
  {
    LOG_ERR("qdrant is ofline.\n");
  }

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return success;
}

std::string qdrant_get_distance(qdrant_distance_type_t distance)
{
  std::string param;

  switch (distance)
  {
  case DotProduct:
  {
    param.assign("Dot");
    break;
  }
  case Euclid:
  {
    param = "Euclid";
    break;
  }
  case Manhattan:
  {
    param = "Manhattan";
    break;
  }
  case Cosine:
  {
    param = "Cosine";
    break;
  }
  default:
  {
    LOG("warning: unknow distance %d,. using default 'Cosine'.\n", distance);
    param = "Cosine";
    break;
  }
  }

  return param;
}

/***
 *** CURL related
 ***/
int qdrant_curl_callback_nop(char *, size_t s, size_t n, void *)
{
  return s * n;
}

int qcc_read_data(char *buffer, size_t size, size_t nmemb, void *userdata)
{
  const size_t total = size * nmemb;
  const char *udata = reinterpret_cast<const char *>(userdata);
  if (NULL == udata)
  {
    return 0;
  }

  LOG("size is %ld, nmemb is %ld\n", size, nmemb);
  LOG("userdata is %p | %s\n", userdata, udata);
  memcpy(buffer, udata, total);

  return total;
}

int qdrant_curl_write_data(char *buffer, size_t size, size_t nmemb,
                           void *userdata)
{
  curl_write_data_t *data = reinterpret_cast<curl_write_data_t *>(userdata);
  if (NULL == data)
  {
    return 0;
  }

  size_t total = size * nmemb;
  if (data->pointer_len < total)
  {
    // Must realloc data
    data->pointer_len = total;
    data->pointer = realloc(data->pointer, total);
  }
  else
  {
    // pointer_len >= total
    data->pointer_len = total;
  }

  memset(data->pointer, 0x00, data->pointer_len);
  memcpy(data->pointer, buffer, total);

  return total;
}

bool qdrant_collection_create(const qdrant_info_t &info,
                              const qdrant_colection_info_t &col)
{
  bool success = true;

  std::string url;
  std::string param(QDRANT_COLLECTIONS_PATH);

  string_replace_all(param, "{collection_name}", col.name);

  url.assign(info.URI);
  url.append(param);

  CURLcode res = curl_global_init(CURL_GLOBAL_ALL);
  if (res != CURLE_OK)
  {
    return false;
  }

  CURL *curl = curl_easy_init();
  if (NULL == curl)
  {
    LOG_ERR("curl_easy_init failed.\n");
    return false;
  }

  nlohmann::json put_data;
  nlohmann::json collection_spec;

  collection_spec["size"] = 1024;
  collection_spec["distance"] = "Cosine";

  put_data["vectors"]["size"] = col.size;
  put_data["vectors"]["distance"] = qdrant_get_distance(col.distance);

  std::string data = nlohmann::to_string(put_data);

  LOG("sending JSON '%s'.\n", data.c_str());
  LOG("to %s\n", url.c_str());

  // Setup curl
  {
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    // CURLOPT_UPLOAD for CURL is Method = 'PUT'
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);

    curl_easy_setopt(curl, CURLOPT_READFUNCTION, qcc_read_data);
    curl_easy_setopt(curl, CURLOPT_READDATA, data.c_str());
    curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t)data.length());

    curl_write_data_t *data = qdrant_malloc_write_data();
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, qdrant_curl_write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, data);

    res = curl_easy_perform(curl);
    if (res != CURLE_OK)
    {
      success = false;
      LOG_ERR("curl_easy_perform failed: %d.\n", res);
    }
    else
    {
      LOG("got result length ... %ld\n", data->pointer_len);
      LOG("got result string ... '%.*s'\n", data->pointer_len,
          (char *)data->pointer);
    }

    free_write_data(data);

    if (headers)
    {
      curl_slist_free_all(headers);
      headers = NULL;
    }
  }

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return success;
}

bool qdrant_collection_delete(const qdrant_info_t &info,
                              const qdrant_colection_info_t &col)
{
  bool success = true;

  std::string url;
  std::string param(QDRANT_COLLECTIONS_PATH);

  string_replace_all(param, "{collection_name}", col.name);

  url.assign(info.URI);
  url.append(param);

  CURLcode res = curl_global_init(CURL_GLOBAL_ALL);
  if (res != CURLE_OK)
  {
    return false;
  }

  CURL *curl = curl_easy_init();
  if (NULL == curl)
  {
    LOG_ERR("curl_easy_init failed.\n");
    return false;
  }

  // Setup curl
  {
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");

    curl_write_data_t *data = qdrant_malloc_write_data();
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, qdrant_curl_write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, data);

    res = curl_easy_perform(curl);
    if (res != CURLE_OK)
    {
      success = false;
      LOG_ERR("curl_easy_perform failed: %d.\n", res);
    }
    else
    {
      LOG("got result length ... %ld\n", data->pointer_len);
      LOG("got result string ... %.*s\n", data->pointer_len,
          (char *)data->pointer);
    }

    free_write_data(data);

    if (headers)
    {
      curl_slist_free_all(headers);
      headers = NULL;
    }
  }

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return success;
  return true;
}

/****************************
 * points API interface
 *****************************/
int qpi_read_data(char *buffer, size_t size, size_t nmemb, void *userdata)
{
  size_t total = size * nmemb;
  if (userdata != NULL)
  {
    memcpy(buffer, userdata, total);
    // LOG("buffer is '%.*s'.\n", (int)total, buffer);
  }
  else
  {
    total = 0;
  }

  return total;
}

bool qdrant_points_insert(const qdrant_info_t &info,
                          const qdrant_colection_info_t &col,
                          const qdrant_point_array_t &points)
{
  nlohmann::json data;
  nlohmann::json itens = nlohmann::json::array();

  for (auto &point : points)
  {
    nlohmann::json item;
    item["id"] = point.id;
    item["payload"][point.payload_x] = point.payload_y;
    item["vector"] = nlohmann::json::array();

    for (auto &vec : point.vector)
    {
      item["vector"].push_back(vec);
    }

    itens.push_back(item);
  }
  data["points"] = itens;

  std::string data_json = nlohmann::to_string(data);
  // LOG_ERR("to send '%s' to API.\n", data_json.c_str());

  CURLcode res = curl_global_init(CURL_GLOBAL_ALL);
  if (res != CURLE_OK)
  {
    LOG_ERR("curl_global_init failed.\n");
    return false;
  }

  CURL *curl = curl_easy_init();
  if (NULL == curl)
  {
    LOG_ERR("curl_easy_init failed.\n");
    curl_global_cleanup();

    return false;
  }
  else
  {
    nlohmann::json result;

    std::string url(info.URI);
    std::string path(QDRANT_POINTS_INSERT_PATH);
    string_replace_all(path, "{collection_name}", col.name);

    url.append(path);

    LOG("sending payload to '%s'.\n", url.c_str());
    LOG("json length is %ld.\n", data_json.length());

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(curl, CURLOPT_READDATA, data_json.c_str());
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, qpi_read_data);
    curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, data_json.length());

    curl_write_data_t *data = qdrant_malloc_write_data();

    curl_easy_setopt(curl, CURLOPT_WRITEDATA, data);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, qdrant_curl_write_data);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    res = curl_easy_perform(curl);
    if (res == CURLE_OK)
    {
      char *json = reinterpret_cast<char *>(data->pointer);
      LOG("got return length ... %ld\n", data->pointer_len);
      LOG("got return string ... %.*s\n", data->pointer_len, json);
    }
    else
    {
      LOG_ERR("curl_easy_perform failed.\n");
    }

    free_write_data(data);

    if (headers != NULL)
    {
      curl_slist_free_all(headers);
    }
  }

  curl_global_cleanup();

  return true;
}
