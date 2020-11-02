/*
 *  This file is part of nifty_gridder.
 *
 *  nifty_gridder is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  nifty_gridder is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with nifty_gridder; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2019-2020 Max-Planck-Society
   Author: Martin Reinecke */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/infra/communication.h"

namespace ducc0 {

namespace detail_pymodule_mpi {

using namespace std;

namespace py = pybind11;

auto None = py::none();

void Pytest_comm(size_t comm_addr)
  {
  MPI_Comm mcomm = *(reinterpret_cast<MPI_Comm *>(comm_addr));
  Communicator comm(mcomm);
  cout << comm.rank() << "/" << comm.num_ranks() << endl;
  }
void Pytest_comm3(size_t hcomm)
  {
  MPI_Comm mcomm = reinterpret_cast<MPI_Comm>(hcomm);
  Communicator comm(mcomm);
  cout << comm.rank() << "/" << comm.num_ranks() << endl;
  }

void add_mpi(py::module &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("mpi");

  m.def("test_comm", &Pytest_comm, "comm_addr"_a);
  m.def("test_comm3", &Pytest_comm3, "comm"_a);
  }

}

using detail_pymodule_mpi::add_mpi;

}
