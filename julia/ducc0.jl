# proposed interface to pass array and type information from Julia to C++

# This code does not work out of the box since I have not updated ducc0_jll yet.
# It should mainly serve as a base for discussion.

# Formatting: format_file("ducc0.jl",indent=2,remove_extra_newlines=true)

module Ducc0

#import ducc0_jll
#const libducc = ducc0_jll.libducc_julia
const libducc = "/home/martin/codes/ducc/julia/ducc_julia.so" # FIXME
#"/home/leoab/OneDrive/UNI/ducc/julia/ducc_julia.so"

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
    return return sizeof(tp(0)) - 1
  elseif tp <: Unsigned
    return return sizeof(tp(0)) - 1 + 32
  elseif tp <: Signed
    return return sizeof(tp(0)) - 1 + 16
  elseif tp == Complex{Float32}
    return typecode(Float32) + 64
  elseif tp == Complex{Float64}
    return typecode(Float64) + 64
  end
end

function Desc(arr::StridedArray{T,N}) where {T,N}
  @assert N <= 10
  # MR the next lines just serve to put shape and stride information into the
  # fixed-size tuples of the descriptor ... is there an easier way to do this?
  shp = zeros(UInt64, 10)
  str = zeros(Int64, 10)
  for i = 1:N
    shp[i] = size(arr)[i]
    str[i] = strides(arr)[i]
  end
  shp = NTuple{10,UInt64}(v for v in shp)
  str = NTuple{10,Int64}(v for v in str)
  # .. up to here

  return ArrayDescriptor(shp, str, pointer(arr), N, typecode(T))
end

Dref = Ref{ArrayDescriptor}

function nufft_best_epsilon(
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
  if res <= 0
    throw(error())
  end
  return res
end

function nufft_u2nu(
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
    Desc(grid),
    Desc(coord),
    0,
    epsilon,
    nthreads,
    Desc(res),
    verbose,
    sigma_min,
    sigma_max,
    periodicity,
    fft_order,
  )
  if ret != 0
    throw(error())
  end
  return res
end

function nufft_nu2u(
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
    Desc(points),
    Desc(coord),
    0,
    epsilon,
    nthreads,
    Desc(res),
    verbose,
    sigma_min,
    sigma_max,
    periodicity,
    fft_order,
  )
  if ret != 0
    throw(error())
  end
  return res
end

mutable struct NufftPlan
  N::Vector{UInt64}
  npoints::Int
  cplan::Ptr{Cvoid}
end

function nufft_delete_plan!(plan::NufftPlan)
  if plan.cplan != C_NULL
    ret = ccall((:nufft_delete_plan, libducc), Cint, (Ptr{Cvoid},), plan.cplan)
    if ret != 0
      throw(error())
    end
    plan.cplan = C_NULL
  end
end

function nufft_make_plan(
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
    Desc(N2),
    Desc(coords),
    epsilon,
    nthreads,
    sigma_min,
    sigma_max,
    periodicity,
    fft_order,
  )

  if ptr == C_NULL
    throw(error())
  end
  p = NufftPlan(N2, size(coords)[2], ptr)
  finalizer(p -> begin
    nufft_delete_plan!(p)
  end, p)

  return p
end

function nufft_nu2u_planned!(
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
    Desc(points),
    Desc(uniform),
  )
  if ret != 0
    throw(error())
  end
end

function nufft_nu2u_planned(
  plan::NufftPlan,
  points::StridedArray{T,1};
  forward::Bool = false,
  verbose::Bool = false,
) where {T}
  res = Array{T}(undef, Tuple(i for i in plan.N))
  nufft_nu2u_planned!(plan, points, res, forward = forward, verbose = verbose)
  return res
end

function nufft_u2nu_planned!(
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
    Desc(uniform),
    Desc(points),
  )
  if ret != 0
    throw(error())
  end
end

function nufft_u2nu_planned(
  plan::NufftPlan,
  uniform::StridedArray{T};
  forward::Bool = true,
  verbose::Bool = false,
) where {T}
  res = Array{T}(undef, plan.npoints)
  nufft_u2nu_planned!(plan, uniform, res, forward = forward, verbose = verbose)
  return res
end

function sht_alm2leg!(
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
    Desc(alm),
    spin,
    lmax,
    Desc(mval),
    Desc(mstart),
    lstride,
    Desc(theta),
    nthreads,
    Desc(leg),
  )
  end
  if ret != 0
    throw(error())
  else
    leg
  end
end

function sht_alm2leg(
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
  sht_alm2leg!(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads)
end

function sht_leg2alm!(
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
    Desc(leg),
    spin,
    lmax,
    Desc(mval),
    Desc(mstart),
    lstride,
    Desc(theta),
    nthreads,
    Desc(alm),
  )
  end
  if ret != 0
    throw(error())
  else
    alm
  end
end

function sht_leg2alm(
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
  nm = length(mval)
  alm = Array{Complex{T}}(undef, (nm, ncomp))
  sht_leg2alm!(leg, alm, spin, lmax, mval, mstart, lstride, theta, nthreads)
end

function sht_leg2map!(
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
    Desc(leg),
    Desc(nphi),
    Desc(phi0),
    Desc(ringstart),
    pixstride,
    nthreads,
    Desc(map),
  )
  end
  if ret != 0
    throw(error())
  else
    map
  end
end

function sht_leg2map(
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
  sht_leg2map!(leg, map, nphi, phi0, ringstart, pixstride, nthreads)
end

function sht_map2leg!(
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
    Desc(map),
    Desc(nphi),
    Desc(phi0),
    Desc(ringstart),
    pixstride,
    nthreads,
    Desc(leg),
  )
  end
  if ret != 0
    throw(error())
  else
    leg
  end
end

function sht_map2leg(
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
  sht_map2leg!(map, leg, nphi, phi0, ringstart, pixstride, nthreads)
end

end

# some demo calls
println(Ducc0.nufft_best_epsilon(UInt64(2), true))
npoints = 1000000
shp = (1000, 1000)
coord = rand(Float64, length(shp), npoints) .- Float32(0.5)
plan = Ducc0.nufft_make_plan(coord, shp)
points = rand(Complex{Float64}, (npoints,))
Ducc0.nufft_nu2u_planned(plan, points)
grid = ones(Complex{Float64}, shp)
Ducc0.nufft_u2nu_planned(plan, grid)
coord = rand(Float32, length(shp), npoints) .- Float32(0.5)
plan = Ducc0.nufft_make_plan(coord, shp)
points = rand(Complex{Float32}, (npoints,))
Ducc0.nufft_nu2u_planned(plan, points)
grid = ones(Complex{Float32}, shp)
Ducc0.nufft_u2nu_planned(plan, grid)
Ducc0.nufft_u2nu_planned!(plan, grid, points)
Ducc0.nufft_nu2u_planned!(plan, points, grid)
Ducc0.nufft_nu2u(coord, points, shp)
Ducc0.nufft_u2nu(coord, grid)
