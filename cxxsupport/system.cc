#include <regex>
#include <string>
#include <fstream>
#include <sstream>

#include "system.h"
#include "string_utils.h"

using namespace std;

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
  myassert (it!=it_end, "did not find pattern '", pattern, "'");
  return stringToData<T>(it->str(1));
  }

} // unnamed namespace

tsize getProcessInfo(const string &quantity)
  {
  string text = fileToString("/proc/self/status");
  return find<tsize>(text, quantity + R"(:\s+(\d+))");
  }

tsize getMemInfo(const string &quantity)
  {
  string text = fileToString("/proc/meminfo");
  return find<tsize>(text, quantity + R"(:\s+(\d+) kB)");
  }

tsize usable_memory()
  {
  string text = fileToString("/proc/meminfo");
  tsize MemTotal = find<tsize>(text, R"(MemTotal:\s+(\d+) kB)");
  tsize Committed = find<tsize>(text, R"(Committed_AS:\s+(\d+) kB)");
  return MemTotal-Committed;
  }
