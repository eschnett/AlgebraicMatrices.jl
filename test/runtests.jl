using AlgebraicMatrices
using Random
using Test

# Set reproducible random number seed
Random.seed!(0)

# Random rationals
function Base.rand(rng::AbstractRNG,
                   ::Random.SamplerType{Rational{T}}) where {T}
    return Rational{T}(T(rand(rng, -1000:1000)) // 1000)
end

@testset "Algebraic vectors" begin
    T = Rational{BigInt}
    for iter in 1:100
        sz = rand(0:10)
        x = avrand(AlgebraicVector{T}, sz)
        y = avrand(AlgebraicVector{T}, sz)
        z = avrand(AlgebraicVector{T}, sz)
        n = zero(AlgebraicVector{T}, sz)
        a = rand(T)
        b = rand(T)

        @test x isa AlgebraicVector{T}
        @test y isa AlgebraicVector{T}
        @test z isa AlgebraicVector{T}
        @test n isa AlgebraicVector{T}
        @test a isa T
        @test b isa T

        @test size(y) == size(x)
        @test size(n) == size(x)

        @test x == x
        @test x == n || x + x ≠ x

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

        @test iszero(evaluate(n))
        @test evaluate(x + y) == evaluate(x) + evaluate(y)
        @test evaluate(a * x) == a * evaluate(x)
        @test evaluate(-x) == -evaluate(x)
    end
end

@testset "Algebraic matrices" begin
    T = Rational{BigInt}
    for iter in 1:100
        sz1 = rand(0:10)
        sz2 = rand(0:10)
        x = avrand(AlgebraicMatrix{T}, sz1, sz2)
        y = avrand(AlgebraicMatrix{T}, sz1, sz2)
        z = avrand(AlgebraicMatrix{T}, sz1, sz2)
        n = zero(AlgebraicMatrix{T}, sz1, sz2)
        a = rand(T)
        b = rand(T)

        @test x isa AlgebraicMatrix{T}
        @test y isa AlgebraicMatrix{T}
        @test z isa AlgebraicMatrix{T}
        @test n isa AlgebraicMatrix{T}
        @test a isa T
        @test b isa T

        @test size(y) == size(x)
        @test size(n) == size(x)

        @test x == x
        @test x == n || x + x ≠ x

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

        @test iszero(evaluate(n))
        @test evaluate(x + y) == evaluate(x) + evaluate(y)
        @test evaluate(a * x) == a * evaluate(x)
        @test evaluate(-x) == -evaluate(x)
    end
end

@testset "Algebraic matrix products" begin
    T = Rational{BigInt}
    for iter in 1:100
        sz1 = rand(0:10)
        sz2 = rand(0:10)
        sz3 = rand(0:10)
        sz4 = rand(0:10)
        x = avrand(AlgebraicMatrix{T}, sz1, sz2)
        x′ = avrand(AlgebraicMatrix{T}, sz1, sz2)
        y = avrand(AlgebraicMatrix{T}, sz2, sz3)
        z = avrand(AlgebraicMatrix{T}, sz3, sz4)
        e1 = one(AlgebraicMatrix{T}, sz1)
        e2 = one(AlgebraicMatrix{T}, sz2)
        nx = zero(AlgebraicMatrix{T}, sz1, sz2)
        ny = zero(AlgebraicMatrix{T}, sz2, sz3)
        nxy = zero(AlgebraicMatrix{T}, sz1, sz3)
        a = rand(T)

        @test (x * y) * z == x * (y * z)
        @test e1 * x == x
        @test x * e2 == x
        @test nx * y == nxy
        @test x * ny == nxy
        @test (a * x) * y == a * (x * y)
        @test x * (a * y) == a * (x * y)
        @test (x + x′) * y == x * y + x′ * y

        @test isone(evaluate(e1))
        @test evaluate(x * y) == evaluate(x) * evaluate(y)
    end
end
