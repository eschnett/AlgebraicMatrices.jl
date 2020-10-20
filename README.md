# Algebraic Matrices

A Julia package for efficient representations of structured matrices

* [GitHub](https://github.com/eschnett/AlgebraicMatrices.jl): Source code repository
* [![GitHub CI](https://github.com/eschnett/AlgebraicMatrices.jl/workflows/CI/badge.svg)](https://github.com/eschnett/AlgebraicMatrices.jl/actions)

## Overview

When linear operators have a particular structure, then representing
them as two-dimensional arrays might be quite inefficient. For
example, if `A[i, j] = x[i] * y[j]`, then storing the elements of
`A[i, j]` requires much more storage than storing both `x[i]` and
`y[j]`, and matrix operations (e.g. multiplication, factorization) are
much more expensive.

Of course, when a matrix has a simple structure such as `A[i, j] =
x[i] * y[j]`, it is obvious what to do. However, the actual structure
can be quite a bit more complex, and can involve sums, scalings,
tensor sum, tensor products, etc. In many cases, are more efficient
(algebraic) representation can still save space and time.

## Description

An "algebraic" matrix (or vector, or tensor), i.e. an
`AlgebraicArray`, can be created in the following ways:

- `WrappedArray` from an `AbstractArray`
- `ZeroArray`: all elements zero
- `OneArray`: the unit matrix
- `ScaledArray`: a scale factor times an `AlgebraicArray`
- `SumArray`: the sum of two other `AlgebraicArray`s
- `ProdutArray`: the product of two other `AlgebraicArray`s

Algebraic arrays support the usual array operations. Some of these
operations might be much cheaper than there standard counterparts
(e.g. matrix products), others might be much more expensive (e.g.
indexing into a `ProductArray`).
