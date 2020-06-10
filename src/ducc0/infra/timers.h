/*
 *  This file is part of the MR utility library.
 *
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2019-2020 Max-Planck-Society
   Authors: Peter Bell, Martin Reinecke */

#ifndef DUCC0_TIMERS_H
#define DUCC0_TIMERS_H

#include <chrono>
#include <string>
#include <map>

#include "ducc0/infra/error_handling.h"

namespace ducc0 {

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

}

#endif
