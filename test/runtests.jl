using AlgebraicMatrices
using Random
using Test

# Set reproducible random number seed
Random.seed!(0)

# Random rationals
function Base.rand(rng::AbstractRNG,
                   ::Random.SamplerType{Rational{T}}) where {T}
    l = lcm(1:10)
    return Rational{T}(T(rand(rng, (-l):l)) // l)
end

@testset "Algebraic matrices as vector spaces D=$D" for D in 0:3
    T = Rational{BigInt}
    niters = 20
    for iter in 1:niters
        print("\r$iter/$niters...")

        sz = rand(0:10, D)
        x = arand(AlgebraicArray{T,D}, sz)
        y = arand(AlgebraicArray{T,D}, sz)
        z = arand(AlgebraicArray{T,D}, sz)
        n = zero(AlgebraicArray{T,D}, sz)
        a = rand(T)
        b = rand(T)

        @test x isa AbstractArray{T,D}
        @test y isa AbstractArray{T,D}
        @test z isa AbstractArray{T,D}
        @test n isa AbstractArray{T,D}
        @test a isa T
        @test b isa T

        @test size(x) == size(y) == size(z) == size(n)

        @test x == x
        @test x == n || x + x ≠ x

        if D == 1
            @test (x < y) + (x == y) + (x > y) == 1
        end

        @test collect(x) isa Array{T,D}
        @test evaluate(x) isa AbstractArray{T,D}
        @test collect(x) == evaluate(x)

        x′ = map(x -> 2x + 1, x)
        @test x′ == map(x -> 2x + 1, evaluate(x))

        @test (x + y) + z == x + (y + z)
        @test n + x == x
        @test x + n == x
        @test x + y == y + x
        @test one(T) * x == x
        @test x * one(T) == x
        @test a * n == n
        @test n * a == n
        if a ≠ zero(T)
            @test inv(a) * (a * x) == x
        end
        @test +x == x
        @test -x == n - x
        @test x + (-x) == n
        @test (-x) + x == n
        @test a * (x + y) == a * x + a * y
        @test (a + b) * x == a * x + b * x

        @test (x == y) == (evaluate(x) == evaluate(y))
        if D == 1
            @test (x < y) == (evaluate(x) < evaluate(y))
            @test (x > y) == (evaluate(x) > evaluate(y))
        end
        @test iszero(evaluate(n))
        @test evaluate(x + y) == evaluate(x) + evaluate(y)
        @test evaluate(a * x) == a * evaluate(x)
        @test evaluate(-x) == -evaluate(x)
    end
    println("\r", " "^40, "\r")
end

@testset "Algebraic matrices as categories D=$D1" for D1 in 1:3
    T = Rational{BigInt}
    niters = 10
    for iter in 1:niters
        print("\r$iter/$niters...")

        D2 = rand(1:3)
        D3 = rand(1:3)
        sz1 = rand(0:10, D1)
        sz2 = (sz1[end], rand(0:10, D2 - 1)...)
        sz3 = (sz2[end], rand(0:10, D3 - 1)...)
        x = arand(AlgebraicArray{T,D1}, sz1)
        x′ = arand(AlgebraicArray{T,D1}, sz1)
        y = arand(AlgebraicArray{T,D2}, sz2)
        z = arand(AlgebraicArray{T,D3}, sz3)
        e1 = one(AlgebraicArray{T,2}, sz1[1], sz1[1])
        e2 = one(AlgebraicArray{T,2}, sz1[end], sz1[end])
        nx = zero(AlgebraicArray{T,D1}, sz1)
        ny = zero(AlgebraicArray{T,D2}, sz2)
        D12 = D1 + D2 - 2
        sz12 = (sz1[1:(end - 1)]..., sz2[2:end]...)
        nxy = zero(AlgebraicArray{T,D12}, sz12)
        a = rand(T)

        if ndims(y) ≥ 2
            @test (x * y) * z == x * (y * z)
        end
        @test e1 * x == x
        @test x * e2 == x
        @test nx * y == nxy
        @test x * ny == nxy
        @test (a * x) * y == a * (x * y)
        @test x * (a * y) == a * (x * y)
        @test (x + x′) * y == x * y + x′ * y

        @test isone(evaluate(e1))

        x′ = evaluate(x)
        y′ = evaluate(y)
        sxy = (size(x)[1:(end - 1)]..., size(y)[2:end]...)
        xy′ = zeros(T, sxy)
        for i in CartesianIndices(size(x)[1:(end - 1)])
            for j in CartesianIndices(size(y)[2:end])
                for k in 1:size(y, 1)
                    xy′[i, j] += x′[i, k] * y′[k, j]
                end
            end
        end
        # @test evaluate(x * y) == evaluate(x) * evaluate(y)
        @test evaluate(x * y) == xy′
    end
    println("\r", " "^40, "\r")
end
