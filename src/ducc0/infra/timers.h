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
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <map>

#include "ducc0/infra/error_handling.h"

namespace ducc0 {

namespace detail_timers {

using namespace std;

class SimpleTimer
  {
  private:
    using clock = chrono::steady_clock;
    clock::time_point starttime;

  public:
    SimpleTimer()
      : starttime(clock::now()) {}
    void reset()
      { starttime = clock::now(); }
    double operator()() const
      { return chrono::duration<double>(clock::now() - starttime).count(); }
  };

class TimerHierarchy
  {
  private:
    using clock = chrono::steady_clock;

    class tstack_node
      {
      private:
        using maptype = map<string,tstack_node>;
        using Tipair = pair<maptype::const_iterator,double>;

      public:
        tstack_node *parent;
        string name;
        double accTime;
        maptype child;

      private:
        double full_acc() const
          {
          double t_own = accTime;
          for (const auto &nd: child)
            t_own += nd.second.full_acc();
          return t_own;
          }

        size_t max_namelen() const
          {
          auto res=name.length();
          for (const auto &ch: child)
            res=max(res,ch.second.max_namelen());
          return res;
          }
        static void floatformat(double val, size_t pre, size_t post, ostream &os)
          {
          size_t fct=1;
          for (size_t i=0; i<post; ++i, fct*=10);
          os << setw(pre) << int(val) << "." << setw(post) << setfill('0')
             << int((val-int(val))*fct+0.5) << setfill(' ');
          }
        static void printline(const string &indent, int twidth, int slen,
          const string &name, double val, double total,
          ostream &os)
          {
          os << indent << "+- " << name << setw(slen+1-name.length()) << ":";
          floatformat(100*val/total, 3, 2, os);
          os << "% (";
          floatformat(val, twidth-5, 4, os);
          os << "s)\n";
          }
        void report(const string &indent, int twidth, int slen, ostream &os) const
          {
          double total=full_acc();
          vector<Tipair> tmp;
          for (auto it=child.cbegin(); it!=child.cend(); ++it)
            tmp.push_back(make_pair(it, it->second.full_acc()));

          if (tmp.size()>0)
            {
            sort(tmp.begin(),tmp.end(),
              [](const Tipair &a, const Tipair &b){ return a.second>b.second; });
            double tsum=0;
            os << indent << "|\n";
            for (unsigned i=0; i<tmp.size(); ++i)
              {
              printline(indent, twidth, slen, tmp[i].first->first, tmp[i].second, total, os);
              (tmp[i].first->second).report(indent+"|  ",twidth,slen,os);
              tsum+=tmp[i].second;
              }
            printline(indent, twidth, slen, "<unaccounted>", total-tsum, total, os);
            if (indent!="") os << indent << "\n";
            }
          }

      public:
        tstack_node(const string &name_, tstack_node *parent_=nullptr)
          : parent(parent_), name(name_), accTime(0.) {}

        void report(ostream &os) const
          {
          auto slen=string("<unaccounted>").size();
          slen = max(slen, max_namelen());

          double total=full_acc();
          os << "\nTotal wall clock time for " << name << ": " << setprecision(4) << total << "s\n";
//          printf("\nTotal wall clock time for '%s': %1.4fs\n",name.c_str(),total);

          int logtime=max(1,int(log10(total)+1));
          report("",logtime+5,slen, os);
          }

        void addTime(double dt)
          { accTime += dt; }

        double add_timings(const string &prefix,
          map<string, double> &res) const
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
      curnode->addTime(chrono::duration<double>(tnow - last_time).count());
      last_time = tnow;
      }

    void push_internal(const string &name)
      {
      auto it=curnode->child.find(name);
      if (it==curnode->child.end())
        {
        MR_assert(name.find(':') == string::npos, "reserved character");
        it = curnode->child.insert(make_pair(name,tstack_node(name, curnode))).first;
        }
      curnode=&(it->second);
      }

  public:
    TimerHierarchy(const string &name="<root>")
      : last_time(clock::now()), root(name, nullptr), curnode(&root) {}
    void push(const string &name)
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
    void poppush(const string &name)
      {
      pop();
      push_internal(name);
      }
    map<string, double> get_timings()
      {
      adjust_time();
      map<string, double> res;
      root.add_timings("root", res);
      return res;
      }
    void report(ostream &os) const
      { ostringstream oss; root.report(oss); os<<oss.str(); }
  };

}

using detail_timers::SimpleTimer;
using detail_timers::TimerHierarchy;

}

#endif
