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
where `Xs` and `Xp` are vectors of length equal to N. Here `s` and `p` will be vectors representing the safety and performance of each player. To get the safety and performance of a single player with index `i`, just do
```julia
(s, p) = f(prodFunc, i, xs, xp)
```
or, equivalently,
```julia
(s, p) = prodFunc(i, xs, xp)
```
where `xs` and `xp` are both scalar values; the outputs `s` and `p` will also be scalar.
"""
struct ProdFunc{N}
    A::SVector{N, Float64}
    α::SVector{N, Float64}
    B::SVector{N, Float64}
    β::SVector{N, Float64}
    θ::SVector{N, Float64}
end

function ProdFunc(
    ;
    n::Int = 2,
    kwargs...
)
    @assert n >= 2 "n must be at least 2"
    A = if haskey(kwargs, :A)
        as_SVector(kwargs[:A], n)
    else
        @SVector fill(10., n)
    end
    α = if haskey(kwargs, :α)
        as_SVector(kwargs[:α], n)
    elseif haskey(kwargs, :alpha)
        as_SVector(kwargs[:alpha], n)
    else
        @SVector fill(0.5, n)
    end
    B = if haskey(kwargs, :B)
        as_SVector(kwargs[:B], n)
    else
        @SVector fill(10., n)
    end
    β = if haskey(kwargs, :β)
        as_SVector(kwargs[:β], n)
    elseif haskey(kwargs, :beta)
        as_SVector(kwargs[:beta], n)
    else
        @SVector fill(0.5, n)
    end
    θ = if haskey(kwargs, :θ)
        as_SVector(kwargs[:θ], n)
    elseif haskey(kwargs, :theta)
        as_SVector(kwargs[:theta], n)
    else
        @SVector fill(0.25, n)
    end
    return ProdFunc(A, α, B, β, θ)
end

ProdFunc(A, α, B, β, θ) = ProdFunc(n=length(A), A=A, α=α, B=B, β=β, θ=θ)

"""
Create a ProdFunc from a pre-existing prodFunc
"""
function ProdFunc(prodFunc::ProdFunc; kwargs...)
    fields = (
        f => (f in keys(kwargs)) ? kwargs[f] : getfield(prodFunc, f)
        for f in fieldnames(ProdFunc)
    )
    return ProdFunc(; fields...)
end


function f(prodFunc::ProdFunc, i::Int, Xs::Number, Xp::Number)
    p = prodFunc.B[i] * Xp^prodFunc.β[i]
    s = prodFunc.A[i] * Xs^prodFunc.α[i] * p^(-prodFunc.θ[i])
    return s, p
end

# call with a vector of inputs [Xs, Xp]
function f(prodFunc::ProdFunc, i::Int, X::AbstractVector)
    return f(prodFunc, i, X[1], X[2])
end

# allow direct calling of prodFunc
(prodFunc::ProdFunc)(i::Int, Xs, Xp) = f(prodFunc, i, Xs, Xp)
(prodFunc::ProdFunc)(i::Int, X) = f(prodFunc, i, X)

function f(prodFunc::ProdFunc, Xs::AbstractVector, Xp::AbstractVector)
    p = prodFunc.B .* Xp.^prodFunc.β
    s = prodFunc.A .* Xs.^prodFunc.α .* p.^(-prodFunc.θ)
    return s, p
end

# call with a matrix of inputs [Xs Xp]
function f(prodFunc::ProdFunc, X::AbstractMatrix)
    return f(prodFunc, X[:, 1], X[:, 2])
end

(prodFunc::ProdFunc)(Xs, Xp) = f(prodFunc, Xs, Xp)
(prodFunc::ProdFunc)(X) = f(prodFunc, X)

function is_symmetric(prodFunc::ProdFunc{N}) where {N}
    all(
        all(x[1] .== x[2:N])
        for x in (prodFunc.A, prodFunc.α, prodFunc.B, prodFunc.β, prodFunc.θ)
    )
end
