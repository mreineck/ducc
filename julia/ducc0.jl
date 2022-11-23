# proposed interface to pass array and type information from Julia to C++
# TODO: when C++ throws an exception, the whole Julia interpreter crashes ...
# would be nice to avoid this and throw a Julia exception instead.

# This code does not work out of the box since I have not updated ducc0_jll yet.
# It should mainly serve as a base for discussion.
module Ducc0

ducclib = "/home/martin/codes/ducc/julia/ducc_julia.so"

struct ArrayDescriptor
  shape::NTuple{10,UInt64}  # length of every axis
  stride::NTuple{10,Int64}  # stride along every axis (in elements)
  data::Ptr{Cvoid}         # pointer to the first array element
  ndim::UInt8              # number of dimensions
  dtype::UInt8             # magic values determining the data type
end

function ArrayDescriptor(arr::StridedArray{T, N}) where {T,N}
    @assert N <= 10
# MR the next lines just serve to put shape and stride information into the
# fixed-size tuples of the descriptor ... is tere an easier way to do this?
    shp = zeros(UInt64,10)
    str = zeros(Int64,10)
    for i in 1:N
        shp[i]=size(arr)[i]
        str[i]=strides(arr)[i]
    end
    shp = NTuple{10,UInt64}(v for v in shp)
    str = NTuple{10,Int64}(v for v in str)
# .. up to here

# MR this should probably be a static variable if such a thing exists
    typedict = Dict(Float32=>68,
                    Float64=>72,
                    Complex{Float32}=>200,
                    Complex{Float64}=>208,
                    UInt64=>40)
    tcode = typedict[T]
    ArrayDescriptor(shp, str, pointer(arr), N, tcode,)
end

# This is the function that should be called by the end user
function nufft_u2nu(coord::StridedArray{T,2}, grid::StridedArray{T2,N};
    forward::Bool = true,
    verbose::Bool = false,
    epsilon::AbstractFloat = 1e-5,
    nthreads::Unsigned = UInt32(1),
    sigma_min::AbstractFloat = 1.1,
    sigma_max::AbstractFloat = 2.6,
    periodicity::AbstractFloat = 2π,
    fft_order::Bool = true) where {T,T2,N}

    res = Vector{T2}(undef,size(coord)[2])
    GC.@preserve coord grid res
      ret=ccall((:nufft_u2nu, ducclib), Cint, (ArrayDescriptor,ArrayDescriptor, Cint, Cdouble, Csize_t, ArrayDescriptor,Csize_t, Cdouble, Cdouble, Cdouble, Cint), ArrayDescriptor(grid), ArrayDescriptor(coord), 0, epsilon, nthreads, ArrayDescriptor(res), verbose, sigma_min, sigma_max, periodicity, fft_order)
    if ret!=0
      throw(error())
    end
    return res
end
function nufft_nu2u(coord::StridedArray{T,2}, points::StridedArray{T2,1}, N::NTuple{D,Int};
    forward::Bool = true,
    verbose::Bool = false,
    epsilon::AbstractFloat = 1e-5,
    nthreads::Unsigned = UInt32(1),
    sigma_min::AbstractFloat = 1.1,
    sigma_max::AbstractFloat = 2.6,
    periodicity::AbstractFloat = 2π,
    fft_order::Bool = true) where {T,T2,D}

    res = Array{T2}(undef,N)
    GC.@preserve coord points res ret=ccall((:nufft_nu2u, ducclib), Cint, (ArrayDescriptor,ArrayDescriptor, Cint, Cdouble, Csize_t, ArrayDescriptor,Csize_t, Cdouble, Cdouble, Cdouble, Cint), ArrayDescriptor(points), ArrayDescriptor(coord), 0, epsilon, nthreads, ArrayDescriptor(res), verbose, sigma_min, sigma_max, periodicity, fft_order)
    if ret!=0
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
  if plan.cplan!=Ptr{Cvoid}(0)
    ret = ccall((:nufft_delete_plan, ducclib), Cint, (Ptr{Cvoid},), plan.cplan)
    if ret!=0
      throw(error())
    end
    plan.cplan = Ptr{Cvoid}(0)
  end
end
function nufft_make_plan(coords::Matrix{T}, N::NTuple{D,Int}; nu2u::Bool=false,
                             epsilon::AbstractFloat=1e-5,
                             nthreads::Unsigned = UInt32(1),
                             sigma_min::AbstractFloat=1.1,
                             sigma_max::AbstractFloat=2.6,
                             periodicity::AbstractFloat = 2π,
                             fft_order::Bool = true) where {T,D}
  N2=Vector{UInt64}(undef,D)
  for i in 1:D
    N2[i] = N[i]
  end
  GC.@preserve N2 coords
  ptr = ccall((:nufft_make_plan, ducclib), Ptr{Cvoid}, 
                (Cint, ArrayDescriptor, ArrayDescriptor, Cdouble, Csize_t, Cdouble, Cdouble, Cdouble, Cint), 
                nu2u, ArrayDescriptor(N2), ArrayDescriptor(coords), epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order)

  if ptr==Ptr{Cvoid}(0)
    throw(error())
  end
  p = NufftPlan(N2, size(coords)[2], ptr)
  finalizer(p -> begin
    nufft_delete_plan!(p)
  end, p)

  return p
end
function nufft_nu2u_planned!(plan::NufftPlan, points::StridedArray{T,1}, uniform::StridedArray{T}; forward::Bool=false, verbose::Bool=false,) where {T}
  GC.@preserve points uniform
  ret = ccall((:nufft_nu2u_planned, ducclib), Cint, (Ptr{Cvoid}, Cint, Csize_t, ArrayDescriptor, ArrayDescriptor), plan.cplan, forward, verbose, ArrayDescriptor(points), ArrayDescriptor(uniform))
  if ret!=0
    throw(error())
  end
end
function nufft_nu2u_planned(plan::NufftPlan, points::StridedArray{T,1}; forward::Bool=false, verbose::Bool=false,) where {T}
  res = Array{T}(undef, Tuple(i for i in plan.N))
  GC.@preserve points res
  ret = ccall((:nufft_nu2u_planned, ducclib), Cint, (Ptr{Cvoid}, Cint, Csize_t, ArrayDescriptor, ArrayDescriptor), plan.cplan, forward, verbose, ArrayDescriptor(points), ArrayDescriptor(res))
  if ret!=0
    throw(error())
  end
  return res
end
function nufft_u2nu_planned!(plan::NufftPlan, uniform::StridedArray{T}, points::StridedArray{T,1}; forward::Bool=true, verbose::Bool=false,) where {T}
  GC.@preserve uniform points
  ret = ccall((:nufft_u2nu_planned, ducclib), Cint, (Ptr{Cvoid}, Cint, Csize_t, ArrayDescriptor, ArrayDescriptor), plan.cplan, forward, verbose, ArrayDescriptor(uniform), ArrayDescriptor(points))
  if ret!=0
    throw(error())
  end
end
function nufft_u2nu_planned(plan::NufftPlan, uniform::StridedArray{T}; forward::Bool=true, verbose::Bool=false,) where {T}
  res = Array{T}(undef, plan.npoints)
  GC.@preserve uniform res
  ret = ccall((:nufft_u2nu_planned, ducclib), Cint, (Ptr{Cvoid}, Cint, Csize_t, ArrayDescriptor, ArrayDescriptor), plan.cplan, forward, verbose, ArrayDescriptor(uniform), ArrayDescriptor(res))
  if ret!=0
    throw(error())
  end
  return res
end

end

# demo call
npoints=1000000
shp=(1000,1000)
coord = rand(Float64, length(shp),npoints) .- Float32(0.5)
plan = Ducc0.nufft_make_plan(coord, shp)
points = rand(Complex{Float64},(npoints,))
Ducc0.nufft_nu2u_planned(plan, points)
grid = ones(Complex{Float64},shp)
Ducc0.nufft_u2nu_planned(plan, grid)
coord = rand(Float32, length(shp),npoints) .- Float32(0.5)
plan = Ducc0.nufft_make_plan(coord, shp)
points = rand(Complex{Float32},(npoints,))
Ducc0.nufft_nu2u_planned(plan, points)
grid = ones(Complex{Float32},shp)
Ducc0.nufft_u2nu_planned(plan, grid)
Ducc0.nufft_u2nu_planned!(plan, grid, points)
Ducc0.nufft_nu2u_planned!(plan, points, grid)
Ducc0.nufft_nu2u(coord, points, shp)
Ducc0.nufft_u2nu(coord, grid)
