#ifndef MRBASE_ERROR_HANDLING_H
#define MRBASE_ERROR_HANDLING_H

#include <iostream>
#include <cstdlib>

#if defined (__GNUC__)
#define LOC_ CodeLocation(__FILE__, __LINE__, __PRETTY_FUNCTION__)
#else
#define LOC_ CodeLocation(__FILE__, __LINE__)
#endif

#define myfail(...) \
  do { \
    if (!abort_in_progress__) \
      { \
      abort_in_progress__ = true; \
      streamDump__(std::cerr, LOC_, "\n", ##__VA_ARGS__, "\n"); \
      killjob__(); \
      } \
    std::exit(1); \
    } while(0)

#define myassert(cond,...) \
  do { \
    if (cond); \
    else { myfail("Assertion failure\n", ##__VA_ARGS__); } \
    } while(0)

// to be replaced with std::source_location once available
class CodeLocation
  {
  private:
    const char *file, *func;
    int line;

  public:
    CodeLocation(const char *file_, int line_, const char *func_=nullptr)
      : file(file_), func(func_), line(line_) {}

    std::ostream &print(std::ostream &os) const;
  };

inline std::ostream &operator<<(std::ostream &os, const CodeLocation &loc)
  { return loc.print(os); }

extern bool abort_in_progress__;
void killjob__();

#if 1
template<typename T>
inline void streamDump__(std::ostream &os, const T& value)
  { os << value; }

template<typename T, typename ... Args>
inline void streamDump__(std::ostream &os, const T& value,
  const Args& ... args)
  {
  os << value;
  streamDump__(os, args...);
  }
#else
// hyper-elegant C++2017 version
template<typename ...Args>
inline void streamDump__(std::ostream &os, Args&&... args)
  { (os << ... << args); }
#endif

#endif
