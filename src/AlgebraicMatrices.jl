module AlgebraicMatrices

using LinearAlgebra
using Random

################################################################################

# - remove WrappedVector
# - allow AbstractVector everywhere
# - add tensor product
# - add matrix inverses
# - add tensor decompositions (LU, QR)
# - remove abstract types
# - provide "shell" type (where all operations are provided by functions)

################################################################################

export AlgebraicVector
abstract type AlgebraicVector{T} <: AbstractVector{T} end
Base.eltype(x::AlgebraicVector{T}) where {T} = T
function Base.convert(::Type{Vector}, x::AlgebraicVector{T}) where {T}
    return convert(Vector{T}, x)
end
function Base.zero(::Type{<:AlgebraicVector{T}}, size::Int) where {T}
    return ZeroVector{T}(size)
end
Base.:+(x::AlgebraicVector) = ScaledVector(+one(eltype(x)), x)
Base.:-(x::AlgebraicVector) = ScaledVector(-one(eltype(x)), x)
Base.:+(x::AlgebraicVector, y::AlgebraicVector) = SumVector(x, y)
Base.:-(x::AlgebraicVector, y::AlgebraicVector) = x + (-y)
Base.:*(a::Number, x::AlgebraicVector) = ScaledVector(a, x)
Base.:*(x::AlgebraicVector, a::Number) = a * x
Base.:\(a::Number, x::AlgebraicVector) = inv(a) * x
Base.:/(x::AlgebraicVector, a::Number) = x * inv(a)

export avrand
function avrand(rng::AbstractRNG, ::Type{AlgebraicVector{T}},
                size::Int) where {T}
    types = [WrappedVector, ZeroVector, ScaledVector, SumVector]
    return avrand(rng, rand(types){T}, size)
end
function avrand(T::Type{<:AlgebraicVector}, size::Int)
    return avrand(Random.GLOBAL_RNG, T, size)
end

function Base.:(==)(x::AlgebraicVector, y::AlgebraicVector)
    size(x) ≠ size(y) && return false
    return evaluate(x) == evaluate(y)
end
function Base.:(<)(x::AlgebraicVector, y::AlgebraicVector)
    return evaluate(x) < evaluate(y)
end

export WrappedVector
struct WrappedVector{T} <: AlgebraicVector{T}
    x::AbstractVector{T}
    WrappedVector{T}(x::AbstractVector{T}) where {T} = new{T}(x)
    WrappedVector(x::AbstractVector{T}) where {T} = new{T}(x)
end
Base.size(x::WrappedVector) = size(x.x)
function Base.convert(::Type{<:Vector{T}}, x::WrappedVector{T}) where {T}
    return convert(Vector{T}, x.x)::Vector{T}
end
export evaluate
evaluate(x::WrappedVector{T}) where {T} = x.x::AbstractVector{T}
function avrand(rng::AbstractRNG, ::Type{WrappedVector{T}}, size::Int) where {T}
    return WrappedVector{T}(rand(T, size))
end

export ZeroVector
struct ZeroVector{T} <: AlgebraicVector{T}
    size::Int
end
Base.size(x::ZeroVector) = (x.size,)
Base.convert(::Type{<:Vector{T}}, x::ZeroVector{T}) where {T} = zeros(T, x.size)
evaluate(x::ZeroVector{T}) where {T} = zeros(T, x.size)::AbstractVector{T}
function avrand(rng::AbstractRNG, ::Type{ZeroVector{T}}, size::Int) where {T}
    return ZeroVector{T}(size)
end

export ScaledVector
struct ScaledVector{T} <: AlgebraicVector{T}
    a::Number
    x::AlgebraicVector
    function ScaledVector{T}(a::Number, x::AlgebraicVector) where {T}
        R = typeof(one(typeof(a)) * zero(eltype(x)))
        @assert R <: T
        return new{T}(a, x)
    end
    function ScaledVector(a::Number, x::AlgebraicVector)
        R = typeof(one(typeof(a)) * zero(eltype(x)))
        return new{R}(a, x)
    end
end
Base.size(x::ScaledVector) = size(x.x)
function Base.convert(::Type{<:Vector{T}}, x::ScaledVector{T}) where {T}
    return (x.a * convert(Vector{T}, x.x))::Vector{T}
end
function evaluate(x::ScaledVector{T}) where {T}
    return (x.a * evaluate(x.x))::AbstractVector{T}
end
function avrand(rng::AbstractRNG, ::Type{ScaledVector{T}}, size::Int) where {T}
    a = rand(rng, T)
    x = avrand(rng, AlgebraicVector{T}, size)
    return ScaledVector{T}(a, x)
end

export SumVector
struct SumVector{T} <: AlgebraicVector{T}
    x::AlgebraicVector
    y::AlgebraicVector
    function SumVector{T}(x::AlgebraicVector, y::AlgebraicVector) where {T}
        R = typeof(zero(eltype(x)) + zero(eltype(y)))
        @assert R <: T
        @assert size(x) == size(y)
        return new{T}(x, y)
    end
    function SumVector(x::AlgebraicVector, y::AlgebraicVector)
        R = typeof(zero(eltype(x)) + zero(eltype(y)))
        @assert size(x) == size(y)
        return new{R}(x, y)
    end
end
Base.size(x::SumVector) = size(x.x)
function Base.convert(::Type{<:Vector{T}}, x::SumVector{T}) where {T}
    return (convert(Vector{T}, x.x) + convert(Vector{T}, x.y))::Vector{T}
end
function evaluate(x::SumVector{T}) where {T}
    return (evaluate(x.x) + evaluate(x.y))::AbstractVector{T}
end
function avrand(rng::AbstractRNG, ::Type{SumVector{T}}, size::Int) where {T}
    x = avrand(rng, AlgebraicVector{T}, size)
    y = avrand(rng, AlgebraicVector{T}, size)
    return SumVector{T}(x, y)
end

################################################################################

export AlgebraicMatrix
abstract type AlgebraicMatrix{T} <: AbstractMatrix{T} end
Base.eltype(x::AlgebraicMatrix{T}) where {T} = T
function Base.convert(::Type{Matrix}, x::AlgebraicMatrix{T}) where {T}
    return convert(Matrix{T}, x)::Matrix{T}
end
function Base.zero(::Type{<:AlgebraicMatrix{T}}, size1::Int,
                   size2::Int) where {T}
    return ZeroMatrix{T}(size1, size2)
end
function Base.one(::Type{<:AlgebraicMatrix{T}}, size::Int) where {T}
    return OneMatrix{T}(size)
end
Base.:+(x::AlgebraicMatrix) = ScaledMatrix(+one(eltype(x)), x)
Base.:-(x::AlgebraicMatrix) = ScaledMatrix(-one(eltype(x)), x)
Base.:+(x::AlgebraicMatrix, y::AlgebraicMatrix) = SumMatrix(x, y)
Base.:-(x::AlgebraicMatrix, y::AlgebraicMatrix) = x + (-y)
Base.:*(a::Number, x::AlgebraicMatrix) = ScaledMatrix(a, x)
Base.:*(x::AlgebraicMatrix, a::Number) = a * x
Base.:\(a::Number, x::AlgebraicMatrix) = inv(a) * x
Base.:/(x::AlgebraicMatrix, a::Number) = x * inv(a)
Base.:*(x::AlgebraicMatrix, y::AlgebraicMatrix) = ProductMatrix(x, y)
Base.:*(x::AlgebraicMatrix, y::AlgebraicVector) = ProductVector(x, y)

function avrand(rng::AbstractRNG, ::Type{AlgebraicMatrix{T}}, size1::Int,
                size2::Int) where {T}
    types = [WrappedMatrix, ZeroMatrix, ScaledMatrix, SumMatrix]
    return avrand(rng, rand(types){T}, size1, size2)
end
function avrand(T::Type{<:AlgebraicMatrix}, size1::Int, size2::Int)
    return avrand(Random.GLOBAL_RNG, T, size1, size2)
end

function Base.:(==)(x::AlgebraicMatrix, y::AlgebraicMatrix)
    size(x) ≠ size(y) && return false
    return evaluate(x) == evaluate(y)
end
function Base.:(<)(x::AlgebraicMatrix, y::AlgebraicMatrix)
    return evaluate(x) < evaluate(y)
end

export WrappedMatrix
struct WrappedMatrix{T} <: AlgebraicMatrix{T}
    x::AbstractMatrix{T}
    WrappedMatrix{T}(x::AbstractMatrix{T}) where {T} = new{T}(x)
    WrappedMatrix(x::AbstractMatrix{T}) where {T} = new{T}(x)
end
Base.size(x::WrappedMatrix) = size(x.x)
function Base.convert(::Type{<:Matrix{T}}, x::WrappedMatrix{T}) where {T}
    return convert(Matrix{T}, x.x)::Matrix{T}
end
evaluate(x::WrappedMatrix{T}) where {T} = x.x::AbstractMatrix{T}
function avrand(rng::AbstractRNG, ::Type{WrappedMatrix{T}}, size1::Int,
                size2::Int) where {T}
    return WrappedMatrix{T}(rand(T, size1, size2))
end

export ZeroMatrix
struct ZeroMatrix{T} <: AlgebraicMatrix{T}
    size1::Int
    size2::Int
end
Base.size(x::ZeroMatrix) = (x.size1, x.size2)
function Base.convert(::Type{<:Matrix{T}}, x::ZeroMatrix{T}) where {T}
    return zeros(T, x.size1, x.size2)
end
function evaluate(x::ZeroMatrix{T}) where {T}
    return zeros(T, x.size1, x.size2)::AbstractMatrix{T}
end
function avrand(rng::AbstractRNG, ::Type{ZeroMatrix{T}}, size1::Int,
                size2::Int) where {T}
    return ZeroMatrix{T}(size1, size2)
end

export OneMatrix
struct OneMatrix{T} <: AlgebraicMatrix{T}
    size::Int
end
Base.size(x::OneMatrix) = (x.size, x.size)
function Base.convert(::Type{<:Matrix{T}}, x::OneMatrix{T}) where {T}
    return diagm(ones(T, x.size))::Matrix{T}
end
function evaluate(x::OneMatrix{T}) where {T}
    return Diagonal(ones(T, x.size))::AbstractMatrix{T}
end

export ScaledMatrix
struct ScaledMatrix{T} <: AlgebraicMatrix{T}
    a::Number
    x::AlgebraicMatrix
    function ScaledMatrix{T}(a::Number, x::AlgebraicMatrix) where {T}
        R = typeof(one(typeof(a)) * zero(eltype(x)))
        @assert R <: T
        return new{T}(a, x)
    end
    function ScaledMatrix(a::Number, x::AlgebraicMatrix)
        R = typeof(one(typeof(a)) * zero(eltype(x)))
        return new{R}(a, x)
    end
end
Base.size(x::ScaledMatrix) = size(x.x)
function Base.convert(::Type{<:Matrix{T}}, x::ScaledMatrix{T}) where {T}
    return (x.a * convert(Matrix{T}, x.x))::Matrix{T}
end
function evaluate(x::ScaledMatrix{T}) where {T}
    return (x.a * evaluate(x.x))::AbstractMatrix{T}
end
function avrand(rng::AbstractRNG, ::Type{ScaledMatrix{T}}, size1::Int,
                size2::Int) where {T}
    a = rand(rng, T)
    x = avrand(rng, AlgebraicMatrix{T}, size1, size2)
    return ScaledMatrix{T}(a, x)
end

export SumMatrix
struct SumMatrix{T} <: AlgebraicMatrix{T}
    x::AlgebraicMatrix
    y::AlgebraicMatrix
    function SumMatrix{T}(x::AlgebraicMatrix, y::AlgebraicMatrix) where {T}
        R = typeof(zero(eltype(x)) + zero(eltype(y)))
        @assert R <: T
        @assert size(x) == size(y)
        return new{T}(x, y)
    end
    function SumMatrix(x::AlgebraicMatrix, y::AlgebraicMatrix)
        R = typeof(zero(eltype(x)) + zero(eltype(y)))
        @assert size(x) == size(y)
        return new{R}(x, y)
    end
end
Base.size(x::SumMatrix) = size(x.x)
function Base.convert(::Type{<:Matrix{T}}, x::SumMatrix{T}) where {T}
    return convert(Matrix{T}, x.x) + convert(Matrix{T}, x.y)::Matrix{T}
end
function evaluate(x::SumMatrix{T}) where {T}
    return (evaluate(x.x) + evaluate(x.y))::AbstractMatrix{T}
end
function avrand(rng::AbstractRNG, ::Type{SumMatrix{T}}, size1::Int,
                size2::Int) where {T}
    x = avrand(rng, AlgebraicMatrix{T}, size1, size2)
    y = avrand(rng, AlgebraicMatrix{T}, size1, size2)
    return SumMatrix{T}(x, y)
end

export ProductMatrix
struct ProductMatrix{T} <: AlgebraicMatrix{T}
    x::AlgebraicMatrix
    y::AlgebraicMatrix
    function ProductMatrix{T}(x::AlgebraicMatrix, y::AlgebraicMatrix) where {T}
        R = typeof(one(eltype(x)) * one(eltype(y)))
        @assert R <: T
        @assert size(x, 2) == size(y, 1)
        return new{T}(x, y)
    end
    function ProductMatrix(x::AlgebraicMatrix, y::AlgebraicMatrix)
        R = typeof(one(eltype(x)) * one(eltype(y)))
        @assert size(x, 2) == size(y, 1)
        return new{R}(x, y)
    end
end
Base.size(x::ProductMatrix) = (size(x.x, 1), size(x.y, 2))
function Base.convert(::Type{<:Matrix{T}}, x::ProductMatrix{T}) where {T}
    return convert(Matrix{T}, x.x) * convert(Matrix{T}, x.y)::Matrix{T}
end
function evaluate(x::ProductMatrix{T}) where {T}
    return (evaluate(x.x) * evaluate(x.y))::AbstractMatrix{T}
end

export ProductVector
struct ProductVector{T} <: AlgebraicVector{T}
    x::AlgebraicMatrix
    y::AlgebraicVector
    function ProductVector{T}(x::AlgebraicMatrix, y::AlgebraicVector) where {T}
        R = typeof(one(eltype(x)) * one(eltype(y)))
        @assert R <: T
        @assert size(x, 2) == size(y, 1)
        return new{T}(x, y)
    end
    function ProductVector(x::AlgebraicMatrix, y::AlgebraicVector)
        R = typeof(one(eltype(x)) * one(eltype(y)))
        @assert size(x, 2) == size(y, 1)
        return new{R}(x, y)
    end
end
Base.size(x::ProductVector) = (size(x.x, 1),)
function Base.convert(::Type{<:Vector{T}}, x::ProductVector{T}) where {T}
    return convert(Matrix{T}, x.x) * convert(Vector{T}, x.y)::Vector{T}
end
function evaluate(x::ProductVector{T}) where {T}
    return (evaluate(x.x) * evaluate(x.y))::AbstractVector{T}
end

end
