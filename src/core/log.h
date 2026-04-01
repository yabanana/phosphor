#pragma once

#include <cstdio>
#include <ctime>

#ifndef PHOSPHOR_LOG_LEVEL
  #ifdef NDEBUG
    #define PHOSPHOR_LOG_LEVEL 1
  #else
    #define PHOSPHOR_LOG_LEVEL 0
  #endif
#endif

#define PHOSPHOR_LOG_(level_tag, level_num, fmt, ...)                          \
    do {                                                                       \
        if constexpr (level_num >= PHOSPHOR_LOG_LEVEL) {                       \
            std::time_t t_ = std::time(nullptr);                               \
            char ts_[20];                                                      \
            std::strftime(ts_, sizeof(ts_), "%Y-%m-%d %H:%M:%S",              \
                          std::localtime(&t_));                                \
            std::fprintf(stderr, "[%s] [%s] %s:%d: " fmt "\n",               \
                         level_tag, ts_, __FILE__, __LINE__, ##__VA_ARGS__);   \
        }                                                                      \
    } while (0)

#if PHOSPHOR_LOG_LEVEL <= 0
  #define LOG_DEBUG(fmt, ...) PHOSPHOR_LOG_("DEBUG", 0, fmt, ##__VA_ARGS__)
#else
  #define LOG_DEBUG(fmt, ...) ((void)0)
#endif

#define LOG_INFO(fmt, ...)  PHOSPHOR_LOG_("INFO",  1, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  PHOSPHOR_LOG_("WARN",  2, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) PHOSPHOR_LOG_("ERROR", 3, fmt, ##__VA_ARGS__)
