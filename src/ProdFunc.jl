@doc raw"""
The `ProdFunc.jl` file implements a `ProdFunc` (production function) type that contains the variables for the function

$$f(X_s, X_p) = (AX_s^\alpha (BX_p^\beta)^{-\theta},\ BX_p^\beta)$$

describing how the inputs *X<sub>s</sub>* and *X<sub>p</sub>* produce safety and performance for all players. You can create an instance of the `ProdFunc` type like
```julia
prodFunc = ProdFunc(
    n = 2,
    A = 10.,
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = 0.25
)
```
If you want to supply different parameters for each player or just like typing more stuff out, you can also supply the parameters as vectors of equal length (equal to `n`). The following creates the same object as the example above:
```julia
prodFunc = ProdFunc(
    n = 2,
    A = [10., 10.],
    α = [0.5, 0.5],
    B = [10., 10.],
    β = [0.5, 0.5],
    θ = [0.25, 0.25]
)
```
If you omit a keyword argument, it will be set at a default value.

To determine the outputs for all players, you can do
```julia
(s, p) = f(prodFunc, Xs, Xp)
```
or, equivalently,
```julia
(s, p) = prodFunc(Xs, Xp)
```
where `Xs` and `Xp` are vectors of length equal to `prodFunc.n`. Here `s` and `p` will be vectors representing the safety and performance of each player. To get the safety and performance of a single player with index `i`, just do
```julia
(s, p) = f(prodFunc, i, xs, xp)
```
or, equivalently,
```julia
(s, p) = prodFunc(i, xs, xp)
```
where `xs` and `xp` are both scalar values; the outputs `s` and `p` will also be scalar.
"""
struct ProdFunc{T <: Real}
    n::Int
    A::Vector{T}
    α::Vector{T}
    B::Vector{T}
    β::Vector{T}
    θ::Vector{T}
end

function ProdFunc(
    ;
    n::Int = 2,
    A::Union{Real, AbstractVector} = 10.,
    α::Union{Real, AbstractVector} = 0.5,
    B::Union{Real, AbstractVector} = 10.,
    β::Union{Real, AbstractVector} = 0.5, 
    θ::Union{Real, AbstractVector} = 0.5
)
    @assert n >= 2 "n must be at least 2"
    prodFunc = ProdFunc(
        n,
        as_Float64_Array(A, n),
        as_Float64_Array(α, n),
        as_Float64_Array(B, n),
        as_Float64_Array(β, n),
        as_Float64_Array(θ, n)
    )
    @assert all(length(getfield(prodFunc, x)) == n for x in [:A, :α, :B, :β, :θ]) "Your input params need to match the number of players"
    return prodFunc
end

ProdFunc(A, α, B, β, θ) = ProdFunc(n=length(A), A=A, α=α, B=B, β=β, θ=θ)


function f(prodFunc::ProdFunc, i::Int, Xs::Number, Xp::Number)
    p = prodFunc.B[i] * Xp^prodFunc.β[i]
    s = prodFunc.A[i] * Xs^prodFunc.α[i] * p^(-prodFunc.θ[i])
    return s, p
end

# allow direct calling of prodFunc
(prodFunc::ProdFunc)(i::Int, Xs::Number, Xp::Number) = f(prodFunc, i, Xs, Xp)

function f(prodFunc::ProdFunc, Xs::Vector, Xp::Vector)
    p = prodFunc.B .* Xp.^prodFunc.β
    s = prodFunc.A .* Xs.^prodFunc.α .* p.^(-prodFunc.θ)
    return s, p
end

(prodFunc::ProdFunc)(Xs::Vector, Xp::Vector) = f(prodFunc, Xs, Xp)

function is_symmetric(prodFunc::ProdFunc)
    all(
        all(x[1] .== x[2:prodFunc.n])
        for x in (prodFunc.A, prodFunc.α, prodFunc.B, prodFunc.β, prodFunc.θ)
    )
end
