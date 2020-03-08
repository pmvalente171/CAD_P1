
#ifndef __WB_LOGGER_H__
#define __WB_LOGGER_H__

#include <iostream>

#define FILE__ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : (__FILE__))


#define wbLog(level, ...) {  std::cerr  << "[" << FILE__ << ":" \
						<<  __LINE__ << "]: " <<  wbString(__VA_ARGS__) << std::endl; }\


#endif /* __WB_LOGGER_H__ */
