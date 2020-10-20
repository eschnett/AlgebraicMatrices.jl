module AlgebraicMatrices

using LinearAlgebra
using Random

################################################################################

# - add tensor product
# - add matrix inverses
# - add tensor decompositions (LU, QR)
# - provide "shell" type (where all operations are provided by functions)

################################################################################

export AlgebraicArray
abstract type AlgebraicArray{T,D} <: AbstractArray{T,D} end

const AlgebraicVector{T} = AlgebraicArray{T,1}
const AlgebraicMatrix{T} = AlgebraicArray{T,2}

################################################################################

@inline function sizes2tuple(::Val{D}, sizes...) where {D}
    if length(sizes) == 1
        # if `sizes` is a single argument, it can either be an integer
        # or a collection
        sizes = sizes[1]
        if sizes isa Integer
            sizes = (sizes,)
        end
    end
    length(sizes) == D || error("ndims mismatch")
    sizes = ntuple(d -> Int(sizes[d]), D)
    return sizes::NTuple{D,Int}
end

################################################################################

Base.eltype(x::AlgebraicArray{T}) where {T} = T

function Base.zero(::Type{<:AlgebraicArray{T,D}}, sizes...) where {T,D}
    return ZeroArray{T,D}(sizes2tuple(Val(D), sizes...))
end
Base.:+(x::AlgebraicArray) = ScaledArray(+one(eltype(x)), x)
Base.:-(x::AlgebraicArray) = ScaledArray(-one(eltype(x)), x)
Base.:+(x::AlgebraicArray, y::AlgebraicArray) = SumArray(x, y)
Base.:+(x::AlgebraicArray, y::AbstractArray) = SumArray(x, y)
Base.:+(x::AbstractArray, y::AlgebraicArray) = SumArray(x, y)
Base.:-(x::AlgebraicArray, y::AlgebraicArray) = x + (-y)
Base.:-(x::AlgebraicArray, y::AbstractArray) = x + (-y)
Base.:-(x::AbstractArray, y::AlgebraicArray) = x + (-y)
Base.:*(a::Number, x::AlgebraicArray) = ScaledArray(a, x)
Base.:*(x::AlgebraicArray, a::Number) = a * x
Base.:\(a::Number, x::AlgebraicArray) = inv(a) * x
Base.:/(x::AlgebraicArray, a::Number) = x * inv(a)

function Base.one(::Type{<:AlgebraicArray{T,2}}, sizes...) where {T}
    sizes = sizes2tuple(Val(2), sizes...)
    sizes[1] == sizes[2] || error("size error")
    return OneArray{T}(sizes[1])
end
Base.:*(x::AlgebraicArray, y::AlgebraicArray) = ProductArray(x, y)
Base.:*(x::AlgebraicArray, y::AbstractArray) = ProductArray(x, y)
Base.:*(x::AbstractArray, y::AlgebraicArray) = ProductArray(x, y)
function Base.:*(x::AlgebraicArray{T,1} where {T},
                 y::AlgebraicArray{T,2} where {T})
    return ProductArray(x, y)
end
function Base.:*(x::AlgebraicArray{T,2} where {T},
                 y::AlgebraicArray{T,1} where {T})
    return ProductArray(x, y)
end
function Base.:*(x::AbstractArray{T,1} where {T},
                 y::AlgebraicArray{T,2} where {T})
    return ProductArray(x, y)
end

################################################################################

export arand
function arand(rng::AbstractRNG, ::Type{AlgebraicArray{T,D}}, sizes...;
               maxdepth=nothing) where {T,D}
    if maxdepth ≡ nothing
        maxdepth = 4
    end
    types = [WrappedArray{T,D}, ZeroArray{T,D}]
    if D == 2
        sz = sizes2tuple(Val(2), sizes...)
        if sz[1] == sz[2]
            push!(types, OneArray{T})
        end
    end
    if maxdepth > 0
        append!(types, [ScaledArray{T,D}, SumArray{T,D}, ProductArray{T,D}])
    end
    return arand(rng, rand(types), sizes...; maxdepth=maxdepth - 1)
end
function arand(T::Type, sizes...; maxdepth=nothing)
    return arand(Random.GLOBAL_RNG, T, sizes...; maxdepth=maxdepth)
end

################################################################################

function Base.:(==)(x::AlgebraicArray, y::AlgebraicArray)
    x ≡ y && return true
    size(x) ≠ size(y) && return false
    @inbounds for i in eachindex(x)
        x[i] ≠ y[i] && return false
    end
    return true
end
function Base.:(<)(x::AlgebraicVector, y::AlgebraicVector)
    x ≡ y && return false
    lx = length(x)
    ly = length(y)
    @inbounds for i in 1:min(lx, ly)
        c = cmp(x[i], y[i])
        c ≠ 0 && return c < 0
    end
    return lx < ly
end

################################################################################

export evaluate
evaluate(x::AbstractArray) = x
function arand(rng::AbstractRNG, ::Type{Array{T,D}}, sizes...;
               maxdepth=nothing) where {T,D} end

################################################################################

export WrappedArray
struct WrappedArray{T,D} <: AlgebraicArray{T,D}
    x::AbstractArray{T,D}
end
Base.getindex(x::WrappedArray, inds...) = x.x[inds...]
Base.size(x::WrappedArray) = size(x.x)
evaluate(x::WrappedArray) = x.x
function arand(rng::AbstractRNG, ::Type{WrappedArray{T,D}}, sizes...;
               maxdepth=nothing) where {T,D}
    sz = sizes2tuple(Val(D), sizes...)
    if D == 0
        return WrappedArray{T,D}(fill(rand(T), sz))
    end
    return WrappedArray{T,D}(rand(T, sz...))
end

################################################################################

export ZeroArray
struct ZeroArray{T,D} <: AlgebraicArray{T,D}
    sizes::NTuple{D,Int}
end
Base.getindex(x::ZeroArray, inds...) = zero(eltype(x))
Base.size(x::ZeroArray) = x.sizes
evaluate(x::ZeroArray{T,D}) where {T,D} = zeros(T, x.sizes)::AbstractArray{T,D}
function arand(rng::AbstractRNG, ::Type{ZeroArray{T,D}}, sizes...;
               maxdepth=nothing) where {T,D}
    return ZeroArray{T,D}(sizes2tuple(Val(D), sizes...))
end

################################################################################

export OneArray
struct OneArray{T} <: AlgebraicArray{T,2}
    size::Int
end
function Base.getindex(x::OneArray, inds...)
    inds = sizes2tuple(Val(2), inds...)
    return inds[1] == inds[2] ? one(eltype(x)) : zero(eltype(x))
end
Base.size(x::OneArray) = (x.size, x.size)
evaluate(x::OneArray{T}) where {T} = Diagonal{T}(ones(T, x.size))
function arand(rng::AbstractRNG, ::Type{OneArray{T}}, sizes...;
               maxdepth=nothing) where {T}
    sizes = sizes2tuple(Val(2), sizes...)
    sizes[1] == sizes[2] || error("size error")
    return OneArray{T}(sizes[1])
end

################################################################################

export ScaledArray
struct ScaledArray{T,D} <: AlgebraicArray{T,D}
    a::Number
    x::AbstractArray{U,D} where {U}
    function ScaledArray{T,D}(a::Number,
                              x::AbstractArray{U,D} where {U}) where {T,D}
        R = typeof(one(typeof(a)) * zero(eltype(x)))
        R <: T || error("type mismatch")
        return new{T,D}(a, x)
    end
    function ScaledArray{T}(a::Number,
                            x::AbstractArray{U,D} where {U}) where {T,D}
        return ScaledArray{T,D}(a, x)
    end
    function ScaledArray(a::Number, x::AbstractArray{U,D} where {U}) where {D}
        T = typeof(one(typeof(a)) * zero(eltype(x)))
        return ScaledArray{T,D}(a, x)
    end
end
Base.getindex(x::ScaledArray, inds...) = x.a * x.x[inds...]
Base.size(x::ScaledArray) = size(x.x)
function evaluate(x::ScaledArray{T,D}) where {T,D}
    return (x.a * evaluate(x.x))::AbstractArray{T,D}
end
function arand(rng::AbstractRNG, ::Type{ScaledArray{T,D}}, sizes...;
               maxdepth=nothing) where {T,D}
    a = rand(rng, T)
    x = arand(rng, AlgebraicArray{T,D}, sizes...; maxdepth=maxdepth)
    return ScaledArray{T,D}(a, x)
end

################################################################################

export SumArray
struct SumArray{T,D} <: AlgebraicArray{T,D}
    x::AbstractArray{U,D} where {U}
    y::AbstractArray{U,D} where {U}
    function SumArray{T,D}(x::AbstractArray{U,D} where {U},
                           y::AbstractArray{V,D} where {V}) where {T,D}
        R = typeof(zero(eltype(x)) + zero(eltype(y)))
        R <: T || error("type mismatch")
        size(x) == size(y) || error("size mismatch")
        return new{T,D}(x, y)
    end
    function SumArray{T}(x::AbstractArray{U,D} where {U},
                         y::AbstractArray{V,D} where {V}) where {T,D}
        return SumArray{T,D}(x, y)
    end
    function SumArray(x::AbstractArray{U,D} where {U},
                      y::AbstractArray{V,D} where {V}) where {D}
        T = typeof(zero(eltype(x)) + zero(eltype(y)))
        return SumArray{T,D}(x, y)
    end
end
Base.getindex(x::SumArray, inds...) = x.x[inds...] + x.y[inds...]
Base.size(x::SumArray) = size(x.x)
function evaluate(x::SumArray{T,D}) where {T,D}
    return (evaluate(x.x) + evaluate(x.y))::AbstractArray{T,D}
end
function arand(rng::AbstractRNG, ::Type{SumArray{T,D}}, sizes...;
               maxdepth=nothing) where {T,D}
    sz = sizes2tuple(Val(D), sizes...)
    x = arand(rng, AlgebraicArray{T,D}, sz; maxdepth=maxdepth)
    y = arand(rng, AlgebraicArray{T,D}, sz; maxdepth=maxdepth)
    return SumArray{T,D}(x, y)
end

################################################################################

export ProductArray
struct ProductArray{T,D} <: AlgebraicArray{T,D}
    x::AbstractArray{U,D} where {U,D}
    y::AbstractArray{U,D} where {U,D}
    function ProductArray{T,D}(x::AbstractArray, y::AbstractArray) where {T,D}
        R = typeof(zero(eltype(x)) + zero(eltype(y)))
        R <: T || error("type mismatch")
        (ndims(x) ≥ 1 && ndims(y) ≥ 1) || error("ndims error")
        ndims(x) + ndims(y) == D + 2 || error("ndims mismatch")
        size(x, ndims(x)) == size(y, 1) || error("dimension mismatch")
        return new{T,D}(x, y)
    end
    function ProductArray{T}(x::AbstractArray, y::AbstractArray) where {T}
        D = ndims(x) + ndims(y) - 2
        return ProductArray{T,D}(x, y)
    end
    function ProductArray(x::AbstractArray, y::AbstractArray)
        T = typeof(one(eltype(x)) * one(eltype(y)))
        return ProductArray{T}(x, y)
    end
end
function Base.getindex(x::ProductArray{T,D}, inds...) where {T,D}
    inds = sizes2tuple(Val(D), inds...)
    @assert length(inds) == D
    nd1 = ndims(x.x)
    nd2 = ndims(x.y)
    inds1 = inds[1:(nd1 - 1)]
    inds2 = inds[nd1:D]
    sz′ = size(x.y, 1)
    sz′ == 0 && return zero(T)
    x′ = @view x.x[inds1..., :]
    y′ = @view x.y[:, inds2...]
    return dot(x′, y′)
end
Base.size(x::ProductArray) = (size(x.x)[1:(end - 1)]..., size(x.y)[2:end]...)
function evaluate(x::ProductArray{T,D}) where {T,D}
    sx = size(x.x)
    sy = size(x.y)
    sx′ = (prod(sx[1:(end - 1)]), sx[end])
    sy′ = (sy[1], prod(sy[2:end]))
    return reshape(reshape(evaluate(x.x), sx′) * reshape(evaluate(x.y), sy′),
                   size(x))::AbstractArray{T,D}
end
function arand(rng::AbstractRNG, ::Type{ProductArray{T,D}}, sizes...;
               maxdepth=nothing) where {T,D}
    sz = sizes2tuple(Val(D), sizes...)
    nd1 = rand(1:(D + 1))
    nd2 = D + 2 - nd1
    @assert nd1 ≥ 1 && nd2 ≥ 1 && nd1 + nd2 == D + 2
    sz′ = rand(0:((sz === () ? 0 : maximum(sz)) + 2))
    sz1 = (sz[1:(nd1 - 1)]..., sz′)
    sz2 = (sz′, sz[nd1:(nd1 + nd2 - 2)]...)
    @assert length(sz1) == nd1 && length(sz2) == nd2
    x = arand(rng, AlgebraicArray{T,nd1}, sz1; maxdepth=maxdepth)
    y = arand(rng, AlgebraicArray{T,nd2}, sz2; maxdepth=maxdepth)
    return ProductArray{T,D}(x, y)
end

################################################################################

export simplify
simplify(x::AlgebraicArray) = simplify′(x)

simplify′(x::WrappedArray) = x
simplify′(x::ZeroArray) = x
simplify′(x::OneArray) = x

function simplify′(x::ScaledArray)
    iszero(x.a) && return ZeroArray{eltype(x),ndims(x)}(size(x))
    xx = simplify′(x.x)
    xx isa ZeroArray && return xx
    isone(x.a) && return xx
    xx isa ScaledArray && return ScaledArray(x.a * xx.a, xx.x)
    return ScaledArray(x.a, xx)
end

function simplify′(x::SumArray)
    xx = simplify′(x.x)
    xy = simplify′(x.y)
    xy isa ZeroArray && return xx
    xx isa ZeroArray && return xy
    if xx isa ScaledArray && xy isa ScaledArray && xx.x ≡ xy.x
        a = xx.a + xy.a
        iszero(a) && return ZeroArray{eltype(x),ndims(x)}(size(x))
        isone(a) && return xx.x
        return ScaledArray(xx.a + xy.a, xx.x)
    end
    return rassoc(SumArray(xx, xy))
end
function rassoc(x::SumArray)
    if x.x isa SumArray
        return SumArray(x.x.x, rassoc(SumArray(x.x.y, x.y)))
    end
    return x
end

function simplify′(x::ProductArray)
    xx = simplify′(x.x)
    xy = simplify′(x.y)
    if xx isa ZeroArray || xy isa ZeroArray
        return ZeroArray{eltype(x),ndims(x)}(size(x))
    end
    xy isa OneArray && return xx
    xx isa OneArray && return xy
    if xx isa ScaledArray && xy isa ScaledArray
        a = xx.a * xy.a
        if isone(a)
            xx = xx.x
            xy = xy.x
        else
            xx = ScaledArray(a, xx.x)
            xy = xy.x
        end
    end
    return rassoc(ProductArray(xx, xy))
end
function rassoc(x::ProductArray)
    if x.x isa ProductArray && ndims(x.y) ≥ 2
        return ProductArray(x.x.x, rassoc(ProductArray(x.x.y, x.y)))
    end
    return x
end

end
