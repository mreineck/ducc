#ifndef MRUTIL_TIMERS_H
#define MRUTIL_TIMERS_H

#include <chrono>
#include <string>
#include <map>

#include "mr_util/error_handling.h"
#include "mr_util/string_utils.h"

namespace mr {

namespace timers {

class SimpleTimer
  {
  private:
    using clock = std::chrono::steady_clock;
    clock::time_point starttime;

  public:
    SimpleTimer()
      : starttime(clock::now()) {}
    double operator()() const
      {
      return std::chrono::duration<double>(clock::now() - starttime).count();
      }
  };

class TimerHierarchy
  {
  private:
    using clock = std::chrono::steady_clock;
    class tstack_node
      {
      public:
        tstack_node *parent;
        double accTime;
        std::map<std::string,tstack_node> child;

        tstack_node(tstack_node *parent_)
          : parent(parent_), accTime(0.) {}

        double add_timings(const std::string &prefix,
          std::map<std::string, double> &res) const
          {
          double t_own = accTime;
          for (const auto &nd: child)
            t_own += nd.second.add_timings(prefix+":"+nd.first, res);
          res[prefix] = t_own;
          return t_own;
          }
      };

    clock::time_point last_time;
    tstack_node root;
    tstack_node *curnode;

    void adjust_time()
      {
      auto tnow = clock::now();
      curnode->accTime +=
        std::chrono::duration <double>(tnow - last_time).count();
      last_time = tnow;
      }

    void push_internal(const std::string &name)
      {
      auto it=curnode->child.find(name);
      if (it==curnode->child.end())
        {
        MR_assert(name.find(':') == std::string::npos, "reserved character");
        it = curnode->child.insert(make_pair(name,tstack_node(curnode))).first;
        }
      curnode=&(it->second);
      }

  public:
    TimerHierarchy()
      : last_time(clock::now()), root(nullptr), curnode(&root) {}
    void push(const std::string &name)
      {
      adjust_time();
      push_internal(name);
      }
    void pop()
      {
      adjust_time();
      curnode = curnode->parent;
      MR_assert(curnode!=nullptr, "tried to pop from empty timer stack");
      }
    void poppush(const std::string &name)
      {
      pop();
      push_internal(name);
      }
    std::map<std::string, double> get_timings()
      {
      adjust_time();
      std::map<std::string, double> res;
      root.add_timings("root", res);
      return res;
      }
  };

}}

#endif
