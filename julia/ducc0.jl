# proposed interface to pass array and type information from Julia to C++

# This code does not work out of the box since I have not updated ducc0_jll yet.
# It should mainly serve as a base for discussion.

# Formatting: using JuliaFormatter; format_file("ducc0.jl",indent=2,remove_extra_newlines=true)

module Ducc0

module Support

#import ducc0_jll
#const libducc = ducc0_jll.libducc_julia
const libducc = "/home/martin/codes/ducc/julia/ducc_julia.so" # FIXME
#"/home/martin/codes/ducc/julia/ducc_julia.so"

struct ArrayDescriptor
  shape::NTuple{10,UInt64}  # length of every axis
  stride::NTuple{10,Int64}  # stride along every axis (in elements)
  data::Ptr{Cvoid}          # pointer to the first array element
  ndim::UInt8               # number of dimensions
  dtype::UInt8              # magic values determining the data type
end

# convert data types to type codes for communication with ducc
function typecode(tp::Type)
  if tp <: AbstractFloat
    return sizeof(tp(0)) - 1
  elseif tp <: Unsigned
    return sizeof(tp(0)) - 1 + 32
  elseif tp <: Signed
    return sizeof(tp(0)) - 1 + 16
  elseif tp == Complex{Float32}
    return typecode(Float32) + 64
  elseif tp == Complex{Float64}
    return typecode(Float64) + 64
  end
end

function desc(arr::StridedArray{T,N}) where {T,N}
  @assert N <= 10
  ArrayDescriptor(
    NTuple{10,UInt64}(i <= N ? size(arr)[i] : 0 for i = 1:10),
    NTuple{10,Int64}(i <= N ? strides(arr)[i] : 0 for i = 1:10),
    pointer(arr),
    N,
    typecode(T),
  )
end

Dref = Ref{ArrayDescriptor}

export libducc, desc, Dref

end  # module Support

module Nufft

using ..Support

function best_epsilon(
  ndim::Unsigned,
  singleprec::Bool;
  sigma_min::AbstractFloat = 1.1,
  sigma_max::AbstractFloat = 2.6,
)
  res = ccall(
    (:nufft_best_epsilon, libducc),
    Cdouble,
    (Csize_t, Cint, Cdouble, Cdouble),
    ndim,
    singleprec,
    sigma_min,
    sigma_max,
  )
  res <= 0 && throw(error())
  return res
end

function u2nu(
  coord::StridedArray{T,2},
  grid::StridedArray{T2,N};
  forward::Bool = true,
  verbose::Bool = false,
  epsilon::AbstractFloat = 1e-5,
  nthreads::Unsigned = UInt32(1),
  sigma_min::AbstractFloat = 1.1,
  sigma_max::AbstractFloat = 2.6,
  periodicity::AbstractFloat = 2π,
  fft_order::Bool = true,
) where {T,T2,N}
  res = Vector{T2}(undef, size(coord)[2])
  GC.@preserve coord grid res
  ret = ccall(
    (:nufft_u2nu, libducc),
    Cint,
    (Dref, Dref, Cint, Cdouble, Csize_t, Dref, Csize_t, Cdouble, Cdouble, Cdouble, Cint),
    desc(grid),
    desc(coord),
    0,
    epsilon,
    nthreads,
    desc(res),
    verbose,
    sigma_min,
    sigma_max,
    periodicity,
    fft_order,
  )
  ret != 0 && throw(error())
  return res
end

function nu2u(
  coord::StridedArray{T,2},
  points::StridedArray{T2,1},
  N::NTuple{D,Int};
  forward::Bool = true,
  verbose::Bool = false,
  epsilon::AbstractFloat = 1e-5,
  nthreads::Unsigned = UInt32(1),
  sigma_min::AbstractFloat = 1.1,
  sigma_max::AbstractFloat = 2.6,
  periodicity::AbstractFloat = 2π,
  fft_order::Bool = true,
) where {T,T2,D}
  res = Array{T2}(undef, N)
  GC.@preserve coord points res
  ret = ccall(
    (:nufft_nu2u, libducc),
    Cint,
    (Dref, Dref, Cint, Cdouble, Csize_t, Dref, Csize_t, Cdouble, Cdouble, Cdouble, Cint),
    desc(points),
    desc(coord),
    0,
    epsilon,
    nthreads,
    desc(res),
    verbose,
    sigma_min,
    sigma_max,
    periodicity,
    fft_order,
  )
  ret != 0 && throw(error())
  return res
end

mutable struct NufftPlan
  N::Vector{UInt64}
  npoints::Int
  cplan::Ptr{Cvoid}
end

function delete_plan!(plan::NufftPlan)
  if plan.cplan != C_NULL
    ret = ccall((:nufft_delete_plan, libducc), Cint, (Ptr{Cvoid},), plan.cplan)
    ret != 0 && throw(error())
    plan.cplan = C_NULL
  end
end

function make_plan(
  coords::Matrix{T},
  N::NTuple{D,Int};
  nu2u::Bool = false,
  epsilon::AbstractFloat = 1e-5,
  nthreads::Unsigned = UInt32(1),
  sigma_min::AbstractFloat = 1.1,
  sigma_max::AbstractFloat = 2.6,
  periodicity::AbstractFloat = 2π,
  fft_order::Bool = true,
) where {T,D}
  N2 = Vector{UInt64}(undef, D)
  for i = 1:D
    N2[i] = N[i]
  end
  GC.@preserve N2 coords
  ptr = ccall(
    (:nufft_make_plan, libducc),
    Ptr{Cvoid},
    (Cint, Dref, Dref, Cdouble, Csize_t, Cdouble, Cdouble, Cdouble, Cint),
    nu2u,
    desc(N2),
    desc(coords),
    epsilon,
    nthreads,
    sigma_min,
    sigma_max,
    periodicity,
    fft_order,
  )

  ptr == C_NULL && throw(error())
  p = NufftPlan(N2, size(coords)[2], ptr)
  finalizer(p -> begin
    delete_plan!(p)
  end, p)

  return p
end

function nu2u_planned!(
  plan::NufftPlan,
  points::StridedArray{T,1},
  uniform::StridedArray{T};
  forward::Bool = false,
  verbose::Bool = false,
) where {T}
  GC.@preserve points uniform
  ret = ccall(
    (:nufft_nu2u_planned, libducc),
    Cint,
    (Ptr{Cvoid}, Cint, Csize_t, Dref, Dref),
    plan.cplan,
    forward,
    verbose,
    desc(points),
    desc(uniform),
  )
  ret != 0 && throw(error())
end

function nu2u_planned(
  plan::NufftPlan,
  points::StridedArray{T,1};
  forward::Bool = false,
  verbose::Bool = false,
) where {T}
  res = Array{T}(undef, Tuple(i for i in plan.N))
  nu2u_planned!(plan, points, res, forward = forward, verbose = verbose)
  return res
end

function u2nu_planned!(
  plan::NufftPlan,
  uniform::StridedArray{T},
  points::StridedArray{T,1};
  forward::Bool = true,
  verbose::Bool = false,
) where {T}
  GC.@preserve uniform points
  ret = ccall(
    (:nufft_u2nu_planned, libducc),
    Cint,
    (Ptr{Cvoid}, Cint, Csize_t, Dref, Dref),
    plan.cplan,
    forward,
    verbose,
    desc(uniform),
    desc(points),
  )
  ret != 0 && throw(error())
end

function u2nu_planned(
  plan::NufftPlan,
  uniform::StridedArray{T};
  forward::Bool = true,
  verbose::Bool = false,
) where {T}
  res = Array{T}(undef, plan.npoints)
  u2nu_planned!(plan, uniform, res, forward = forward, verbose = verbose)
  return res
end

end  # module Nufft

module Sht

using ..Support

function alm2leg!(
  alm::StridedArray{Complex{T},2},
  leg::StridedArray{Complex{T},3},
  spin::Unsigned,
  lmax::Unsigned,
  mval::StridedArray{Csize_t,1},
  mstart::StridedArray{Csize_t,1}, # FIXME: this is still 0-based at the moment
  lstride::Int,
  theta::StridedArray{Cdouble,1},
  nthreads::Unsigned,
) where {T}
  GC.@preserve alm mval mstart theta leg begin
    ret = ccall(
      (:sht_alm2leg, libducc),
      Cint,
      (Dref, Csize_t, Csize_t, Dref, Dref, Cptrdiff_t, Dref, Csize_t, Dref),
      desc(alm),
      spin,
      lmax,
      desc(mval),
      desc(mstart),
      lstride,
      desc(theta),
      nthreads,
      desc(leg),
    )
  end
  ret != 0 && throw(error())
  return leg
end

function alm2leg(
  alm::StridedArray{Complex{T},2},
  spin::Unsigned,
  lmax::Unsigned,
  mval::StridedArray{Csize_t,1},
  mstart::StridedArray{Csize_t,1}, # FIXME: this is still 0-based at the moment
  lstride::Int,
  theta::StridedArray{Cdouble,1},
  nthreads::Unsigned,
) where {T}
  ncomp = size(alm, 2)
  ntheta = length(theta)
  nm = length(mval)
  leg = Array{Complex{T}}(undef, (nm, ntheta, ncomp))
  alm2leg!(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads)
  return leg
end

function leg2alm!(
  leg::StridedArray{Complex{T},3},
  alm::StridedArray{Complex{T},2},
  spin::Unsigned,
  lmax::Unsigned,
  mval::StridedArray{Csize_t,1},
  mstart::StridedArray{Csize_t,1}, # FIXME: this is still 0-based at the moment
  lstride::Int,
  theta::StridedArray{Cdouble,1},
  nthreads::Unsigned,
) where {T}
  GC.@preserve leg mval mstart theta alm begin
    ret = ccall(
      (:sht_leg2alm, libducc),
      Cint,
      (Dref, Csize_t, Csize_t, Dref, Dref, Cptrdiff_t, Dref, Csize_t, Dref),
      desc(leg),
      spin,
      lmax,
      desc(mval),
      desc(mstart),
      lstride,
      desc(theta),
      nthreads,
      desc(alm),
    )
  end
  ret != 0 && throw(error())
  return alm
end

function leg2alm(
  leg::StridedArray{Complex{T},3},
  spin::Unsigned,
  lmax::Unsigned,
  mval::StridedArray{Csize_t,1},
  mstart::StridedArray{Csize_t,1}, # FIXME: this is still 0-based at the moment
  lstride::Int,
  theta::StridedArray{Cdouble,1},
  nthreads::Unsigned,
) where {T}
  ncomp = size(leg, 3)
  alm = Array{Complex{T}}(undef, (maximum(mstart) + lmax + 1, ncomp)) # FIXME: still 0-based as well!!
  leg2alm!(leg, alm, spin, lmax, mval, mstart, lstride, theta, nthreads)
end

function leg2map!(
  leg::StridedArray{Complex{T},3},
  map::StridedArray{T,2},
  nphi::StridedArray{Csize_t,1},
  phi0::StridedArray{Cdouble,1},
  ringstart::StridedArray{Csize_t,1}, # FIXME: this is still 0-based at the moment
  pixstride::Int,
  nthreads::Unsigned,
) where {T}
  GC.@preserve leg nphi phi0 ringstart map begin
    ret = ccall(
      (:sht_leg2map, libducc),
      Cint,
      (Dref, Dref, Dref, Dref, Cptrdiff_t, Csize_t, Dref),
      desc(leg),
      desc(nphi),
      desc(phi0),
      desc(ringstart),
      pixstride,
      nthreads,
      desc(map),
    )
  end
  ret != 0 && throw(error())
  return map
end

function leg2map(
  leg::StridedArray{Complex{T},3},
  nphi::StridedArray{Csize_t,1},
  phi0::StridedArray{Cdouble,1},
  ringstart::StridedArray{Csize_t,1}, # FIXME: this is still 0-based at the moment
  pixstride::Int,
  nthreads::Unsigned,
) where {T}
  ncomp = size(leg, 3)
  npix = maximum(ringstart + nphi)
  map = Array{T}(undef, (npix, ncomp))
  leg2map!(leg, map, nphi, phi0, ringstart, pixstride, nthreads)
end

function map2leg!(
  map::StridedArray{T,2},
  leg::StridedArray{Complex{T},3},
  nphi::StridedArray{Csize_t,1},
  phi0::StridedArray{Cdouble,1},
  ringstart::StridedArray{Csize_t,1}, # FIXME: this is still 0-based at the moment
  pixstride::Int,
  nthreads::Unsigned,
) where {T}
  GC.@preserve map nphi phi0 ringstart leg begin
    ret = ccall(
      (:sht_map2leg, libducc),
      Cint,
      (Dref, Dref, Dref, Dref, Cptrdiff_t, Csize_t, Dref),
      desc(map),
      desc(nphi),
      desc(phi0),
      desc(ringstart),
      pixstride,
      nthreads,
      desc(leg),
    )
  end
  ret != 0 && throw(error())
  return leg
end

function map2leg(
  map::StridedArray{T,2},
  nphi::StridedArray{Csize_t,1},
  phi0::StridedArray{Cdouble,1},
  ringstart::StridedArray{Csize_t,1}, # FIXME: this is still 0-based at the moment
  nm::Int,
  pixstride::Int,
  nthreads::Unsigned,
) where {T}
  ncomp = size(map, 2)
  ntheta = length(ringstart)
  leg = Array{Complex{T}}(undef, (nm, ntheta, ncomp))
  map2leg!(map, leg, nphi, phi0, ringstart, pixstride, nthreads)
end

end  # module Sht

end  # module Ducc0

# some demo calls
println(Ducc0.Nufft.best_epsilon(UInt64(2), true))
npoints = 1000000
shp = (1000, 1000)
coord = rand(Float64, length(shp), npoints) .- Float32(0.5)
plan = Ducc0.Nufft.make_plan(coord, shp)
points = rand(Complex{Float64}, (npoints,))
Ducc0.Nufft.nu2u_planned(plan, points)
grid = ones(Complex{Float64}, shp)
Ducc0.Nufft.u2nu_planned(plan, grid)
coord = rand(Float32, length(shp), npoints) .- Float32(0.5)
plan = Ducc0.Nufft.make_plan(coord, shp)
points = rand(Complex{Float32}, (npoints,))
Ducc0.Nufft.nu2u_planned(plan, points)
grid = ones(Complex{Float32}, shp)
Ducc0.Nufft.u2nu_planned(plan, grid)
Ducc0.Nufft.u2nu_planned!(plan, grid, points)
Ducc0.Nufft.nu2u_planned!(plan, points, grid)
Ducc0.Nufft.nu2u(coord, points, shp)
Ducc0.Nufft.u2nu(coord, grid)
