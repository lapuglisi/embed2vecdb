#ifndef __EMBED2VECDB_APP_UTILS_H__
#define __EMBED2VECDB_APP_UTILS_H__

#include <string>
#include <uuid/uuid.h>
#include <vector>

#define LOG(_f, ...) printf("%s: " _f, __func__, ##__VA_ARGS__)
#define LOG_ERR(_f, ...) fprintf(stderr, "%s: " _f, __func__, ##__VA_ARGS__)

std::string generate_uuid(void);

std::vector<std::string> split_lines(const std::string &, const std::string &);

void string_replace_all(std::string &, const std::string &,
                        const std::string &);

#endif // __EMBED2VECDB_APP_UTILS_H__
