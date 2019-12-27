#include <regex>
#include <string>
#include <fstream>
#include <sstream>

#include "mr_util/error_handling.h"
#include "mr_util/system.h"
#include "mr_util/string_utils.h"

using namespace std;
using namespace mr::string_utils;

namespace mr {

namespace system {

namespace {

string fileToString(const string &fname)
  {
  ifstream inp(fname);
  stringstream sbuf;
  sbuf << inp.rdbuf();
  return sbuf.str();
  }

template<typename T> T find(const string &s, const string &pattern)
  {
  regex re(pattern);
  sregex_iterator it(s.begin(), s.end(), re);
  sregex_iterator it_end;
  MR_assert (it!=it_end, "did not find pattern '", pattern, "'");
  return stringToData<T>(it->str(1));
  }

} // unnamed namespace

size_t getProcessInfo(const string &quantity)
  {
  string text = fileToString("/proc/self/status");
  return find<size_t>(text, quantity + R"(:\s+(\d+))");
  }

size_t getMemInfo(const string &quantity)
  {
  string text = fileToString("/proc/meminfo");
  return find<size_t>(text, quantity + R"(:\s+(\d+) kB)");
  }

size_t usable_memory()
  {
  string text = fileToString("/proc/meminfo");
  size_t MemTotal = find<size_t>(text, R"(MemTotal:\s+(\d+) kB)");
  size_t Committed = find<size_t>(text, R"(Committed_AS:\s+(\d+) kB)");
  return MemTotal-Committed;
  }

}}
