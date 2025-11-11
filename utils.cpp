#include "utils.h"
#include <string>
#include <vector>

std::vector<std::string> split_lines(const std::string &source,
                                     const std::string &sep)
{
  std::vector<std::string> lines;
  size_t start = 0;
  size_t end = source.find(sep);

  while (end != std::string::npos)
  {
    lines.push_back(source.substr(start, end - start));
    start = end + sep.length();
    end = source.find(sep, start);
  }

  lines.push_back(source.substr(start)); // Add the last part

  return lines;
}

void string_replace_all(std::string &source, const std::string &find,
                        const std::string &replace)
{
  if (find.empty())
  {
    return;
  }

  std::string builder;
  builder.reserve(source.length());

  size_t pos = 0;
  size_t last_pos = 0;

  while ((pos = source.find(find, last_pos)) != std::string::npos)
  {
    builder.append(source, last_pos, pos - last_pos);
    builder.append(replace);
    last_pos = pos + find.length();
  }
  builder.append(source, last_pos, std::string::npos);

  source = std::move(builder);
}
