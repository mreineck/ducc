#ifndef MRBASE_SYSTEM_H
#define MRBASE_SYSTEM_H

#include <string>
#include "datatypes.h"

tsize getProcessInfo(const std::string &quantity);
tsize getMemInfo(const std::string &quantity);
tsize usable_memory();

#endif
