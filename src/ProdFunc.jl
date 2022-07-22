struct ProdFunc{T <: Real}
    n::Integer
    A::Vector{T}
    α::Vector{T}
    B::Vector{T}
    β::Vector{T}
    θ::Vector{T}
end

ProdFunc(A, α, B, β, θ) = ProdFunc(length(A), A, α, B, β, θ)

function ProdFunc(
    ;
    n::Integer = 2,
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

function f(prodFunc::ProdFunc, i::Integer, Xs::Number, Xp::Number)
    p = prodFunc.B[i] * Xp^prodFunc.β[i]
    s = prodFunc.A[i] * Xs^prodFunc.α[i] * p^(-prodFunc.θ[i])
    return s, p
end

# allow direct calling of prodFunc
(prodFunc::ProdFunc)(i::Integer, Xs::Number, Xp::Number) = f(prodFunc, i, Xs, Xp)

function f(prodFunc::ProdFunc, Xs::Vector, Xp::Vector)
    p = prodFunc.B .* Xp.^prodFunc.β
    s = prodFunc.A .* Xs.^prodFunc.α .* p.^(-prodFunc.θ)
    return s, p
end

(prodFunc::ProdFunc)(Xs::Vector, Xp::Vector) = f(prodFunc, Xs, Xp)

function df_from_s_p(prodFunc::ProdFunc, i::Integer, s::Number, p::Number)
    s_mult = prodFunc.A[i] * prodFunc.α[i] * (s / prodFunc.A[i])^(1. - 1. / prodFunc.α[i])
    p_mult = prodFunc.B[i] * prodFunc.β[i] * (p / prodFunc.B[i]) ^ (1. - 1. / prodFunc.β[i])
    dsdXs = s_mult * p^(-prodFunc.θ[i])
    dsdXp = -prodFunc.θ[i] * s * p^(-prodFunc.θ[i] - 1.) .* p_mult
    return [dsdXs dsdXp; 0. p_mult]
end

function df(prodFunc::ProdFunc, i::Integer, Xs::Number, Xp::Number)
    (s, p) = f(prodFunc, i, Xs, Xp)
    return df_from_s_p(prodFunc, s, p)
end

function df_from_s_p(prodFunc::ProdFunc, s::Vector, p::Vector)
    s_mult = prodFunc.A .* prodFunc.α .* (s ./ prodFunc.A) .^ (1. .- 1. ./ prodFunc.α)
    p_mult = prodFunc.B .* prodFunc.β .* (p ./ prodFunc.B) .^ (1. .- 1. ./ prodFunc.β)
    dsdXs = s_mult .* p.^(-prodFunc.θ)
    dsXp = -prodFunc.θ .* s .* p.^(-prodFunc.θ .- 1.) .* p_mult
    return reshape(
        vcat(dsdXs, dsXp, zeros(size(p_mult)), p_mult),
        prodFunc.n, 2, 2
    )
end

function df(prodFunc::ProdFunc, Xs::Vector, Xp::Vector)
    (s, p) = f(prodFunc, Xs, Xp)
    return df_from_s_p(prodFunc, s, p)
end

function is_symmetric(prodFunc::ProdFunc)
    all(
        all(x[1] .== x[2:prodFunc.n])
        for x in (prodFunc.A, prodFunc.α, prodFunc.B, prodFunc.β, prodFunc.θ)
    )
end
