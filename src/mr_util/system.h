#ifndef MRUTIL_SYSTEM_H
#define MRUTIL_SYSTEM_H

#include <string>
#include <cstdlib>

namespace mr {

namespace system {

std::size_t getProcessInfo(const std::string &quantity);
std::size_t getMemInfo(const std::string &quantity);
std::size_t usable_memory();

}}

#endif
