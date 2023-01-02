# proposed interface to pass array and type information from Julia to C++
# TODO: when C++ throws an exception, the whole Julia interpreter crashes ...
# would be nice to avoid this and throw a Julia exception instead.

# This code does not work out of the box since I have not updated ducc0_jll yet.
# It should mainly serve as a base for discussion.
module Ducc0

#import ducc0_jll
#const libducc = ducc0_jll.libducc_julia
const libducc = "/home/martin/codes/ducc/julia/ducc_julia.so"  # FIXME

struct ArrayDescriptor
  shape::NTuple{10,UInt64}  # length of every axis
  stride::NTuple{10,Int64}  # stride along every axis (in elements)
  data::Ptr{Cvoid}          # pointer to the first array element
  ndim::UInt8               # number of dimensions
  dtype::UInt8              # magic values determining the data type
end

function ArrayDescriptor(arr::StridedArray{T,N}) where {T,N}
  @assert N <= 10
  # MR the next lines just serve to put shape and stride information into the
  # fixed-size tuples of the descriptor ... is tere an easier way to do this?
  shp = zeros(UInt64, 10)
  str = zeros(Int64, 10)
  for i = 1:N
    shp[i] = size(arr)[i]
    str[i] = strides(arr)[i]
  end
  shp = NTuple{10,UInt64}(v for v in shp)
  str = NTuple{10,Int64}(v for v in str)
  # .. up to here

  # MR this should probably be a static variable if such a thing exists
  typedict = Dict(
    Float32 => 68,
    Float64 => 72,
    Complex{Float32} => 200,
    Complex{Float64} => 208,
    UInt64 => 40,
  )
  return ArrayDescriptor(shp, str, pointer(arr), N, typedict[T])
end

function nufft_best_epsilon(
  ndim::Unsigned,
  singleprec::Bool,
  sigma_min::AbstractFloat=1.1,
  sigma_max::AbstractFloat=2.6)

  res = ccall(
    (:nufft_best_epsilon, libducc),
    Cdouble,
    (
      Csize_t,
      Cint,
      Cdouble,
      Cdouble,
    ), ndim, singleprec, sigma_min, sigma_max)
  if res <= 0
    throw(error())
  end
  return res
end

function nufft_u2nu(
  coord::StridedArray{T,2},
  grid::StridedArray{T2,N};
  forward::Bool=true,
  verbose::Bool=false,
  epsilon::AbstractFloat=1e-5,
  nthreads::Unsigned=UInt32(1),
  sigma_min::AbstractFloat=1.1,
  sigma_max::AbstractFloat=2.6,
  periodicity::AbstractFloat=2π,
  fft_order::Bool=true,
) where {T,T2,N}

  res = Vector{T2}(undef, size(coord)[2])
  GC.@preserve coord grid res
  ret = ccall(
    (:nufft_u2nu, libducc),
    Cint,
    (
      Ref{ArrayDescriptor},
      Ref{ArrayDescriptor},
      Cint,
      Cdouble,
      Csize_t,
      Ref{ArrayDescriptor},
      Csize_t,
      Cdouble,
      Cdouble,
      Cdouble,
      Cint,
    ),
    ArrayDescriptor(grid),
    ArrayDescriptor(coord),
    0,
    epsilon,
    nthreads,
    ArrayDescriptor(res),
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
  forward::Bool=true,
  verbose::Bool=false,
  epsilon::AbstractFloat=1e-5,
  nthreads::Unsigned=UInt32(1),
  sigma_min::AbstractFloat=1.1,
  sigma_max::AbstractFloat=2.6,
  periodicity::AbstractFloat=2π,
  fft_order::Bool=true,
) where {T,T2,D}

  res = Array{T2}(undef, N)
  GC.@preserve coord points res
  ret = ccall(
    (:nufft_nu2u, libducc),
    Cint,
    (
      Ref{ArrayDescriptor},
      Ref{ArrayDescriptor},
      Cint,
      Cdouble,
      Csize_t,
      Ref{ArrayDescriptor},
      Csize_t,
      Cdouble,
      Cdouble,
      Cdouble,
      Cint,
    ),
    ArrayDescriptor(points),
    ArrayDescriptor(coord),
    0,
    epsilon,
    nthreads,
    ArrayDescriptor(res),
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
  nu2u::Bool=false,
  epsilon::AbstractFloat=1e-5,
  nthreads::Unsigned=UInt32(1),
  sigma_min::AbstractFloat=1.1,
  sigma_max::AbstractFloat=2.6,
  periodicity::AbstractFloat=2π,
  fft_order::Bool=true,
) where {T,D}
  N2 = Vector{UInt64}(undef, D)
  for i = 1:D
    N2[i] = N[i]
  end
  GC.@preserve N2 coords
  ptr = ccall(
    (:nufft_make_plan, libducc),
    Ptr{Cvoid},
    (
      Cint,
      Ref{ArrayDescriptor},
      Ref{ArrayDescriptor},
      Cdouble,
      Csize_t,
      Cdouble,
      Cdouble,
      Cdouble,
      Cint,
    ),
    nu2u,
    Ref(ArrayDescriptor(N2)),
    Ref(ArrayDescriptor(coords)),
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
  forward::Bool=false,
  verbose::Bool=false,
) where {T}
  GC.@preserve points uniform
  ret = ccall(
    (:nufft_nu2u_planned, libducc),
    Cint,
    (Ptr{Cvoid}, Cint, Csize_t, Ref{ArrayDescriptor}, Ref{ArrayDescriptor}),
    plan.cplan,
    forward,
    verbose,
    Ref(ArrayDescriptor(points)),
    Ref(ArrayDescriptor(uniform)),
  )
  if ret != 0
    throw(error())
  end
end

function nufft_nu2u_planned(
  plan::NufftPlan,
  points::StridedArray{T,1};
  forward::Bool=false,
  verbose::Bool=false,
) where {T}
  res = Array{T}(undef, Tuple(i for i in plan.N))
  nufft_nu2u_planned!(plan, points, res, forward=forward, verbose=verbose)
  return res
end

function nufft_u2nu_planned!(
  plan::NufftPlan,
  uniform::StridedArray{T},
  points::StridedArray{T,1};
  forward::Bool=true,
  verbose::Bool=false,
) where {T}
  GC.@preserve uniform points
  ret = ccall(
    (:nufft_u2nu_planned, libducc),
    Cint,
    (Ptr{Cvoid}, Cint, Csize_t, Ref{ArrayDescriptor}, Ref{ArrayDescriptor}),
    plan.cplan,
    forward,
    verbose,
    Ref(ArrayDescriptor(uniform)),
    Ref(ArrayDescriptor(points)),
  )
  if ret != 0
    throw(error())
  end
end

function nufft_u2nu_planned(
  plan::NufftPlan,
  uniform::StridedArray{T};
  forward::Bool=true,
  verbose::Bool=false,
) where {T}
  res = Array{T}(undef, plan.npoints)
  nufft_u2nu_planned!(plan, uniform, res, forward=forward, verbose=verbose)
  return res
end

function sht_alm2leg(
  alm::StridedArray{T,2},
  spin::Unsigned,
  lmax::Unsigned,
  mval::StridedArray{Csize_t,1},
  mstart::StridedArray{Csize_t,1}, # FIXME: this is still 0-based at the moment
  lstride::Int,
  theta::StridedArray{Cdouble,1},
  nthreads::Unsigned
  ) where{T}
  ncomp = size(alm)[2]
  ntheta = size(theta)[1]
  nm = size(mval)[1]
  res = Array{T}(undef, (nm, ntheta, ncomp))
  GC.@preserve alm mval mstart theta res
  ret = ccall(
    (:sht_alm2leg, libducc),
    Cint,
    (Ref{ArrayDescriptor}, Csize_t, Csize_t, Ref{ArrayDescriptor}, Ref{ArrayDescriptor}, Cptrdiff_t, Ref{ArrayDescriptor}, Csize_t, Ref{ArrayDescriptor}),
    Ref(ArrayDescriptor(alm)),
    spin,
    lmax,
    Ref(ArrayDescriptor(mval)),
    Ref(ArrayDescriptor(mstart)),
    lstride,
    Ref(ArrayDescriptor(theta)),
    nthreads,
    Ref(ArrayDescriptor(leg)))
  if ret != 0
    throw(error())
  end
end

function sht_leg2map(
  leg::StridedArray{Complex{T},3},
  nphi::StridedArray{Csize_t,1},
  phi0::StridedArray{Cdouble,1},
  ringstart::StridedArray{Csize_t,1}, # FIXME: this is still 0-based at the moment
  pixstride::Int,
  nthreads::Unsigned,
  ) where{T}
  ncomp = size(leg)[3]
# FIXME: determine number of pixels as max(nphi+ringstart)
# I don't know how to do this elegantly in Julia
  npix = 42
  res = Array{T}(undef, (npix, ncomp))
  GC.@preserve leg nphi phi0 ringstart res
  ret = ccall(
    (:sht_leg2map, libducc),
    Cint,
    (Ref{ArrayDescriptor}, Ref{ArrayDescriptor}, Ref{ArrayDescriptor}, Ref{ArrayDescriptor}, Cptrdiff_t, Csize_t, Ref{ArrayDescriptor}),
    Ref(ArrayDescriptor(leg)),
    Ref(ArrayDescriptor(nphi)),
    Ref(ArrayDescriptor(phi0)),
    Ref(ArrayDescriptor(ringstart)),
    pixstride,
    nthreads,
    Ref(ArrayDescriptor(res)))
  if ret != 0
    throw(error())
  end
end

end


# some demo calls
println(Ducc0.nufft_best_epsilon(UInt64(2),true))
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
