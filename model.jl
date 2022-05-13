# using Optim

struct ProdFunc
    n::Integer
    A::Vector
    α::Vector
    B::Vector
    β::Vector
    θ::Vector
end

ProdFunc(A, α, B, β, θ) = ProdFunc(length(A), A, α, B, β, θ)

function f(prodFunc::ProdFunc, i::Integer, Xs::Number, Xp::Number)
    p = prodFunc.B[i] * Xp^prodFunc.β[i]
    s = prodFunc.A[i] * Xs^prodFunc.α[i] * p^(-prodFunc.θ[i])
    return s, p
end

function f(prodFunc::ProdFunc, Xs, Xp)
    p = prodFunc.B .* Xp.^prodFunc.β
    s = prodFunc.A .* Xs.^prodFunc.α .* p.^(-prodFunc.θ)
    return s, p
end

function df(prodFunc::ProdFunc, i::Integer, Xs::Number, Xp::Number)
    (s, p) = f(prodFunc, i, Xs, Xp)
    s_mult = prodFunc.A[i] * prodFunc.α[i] * (s / prodFunc.A[i])^(1. - 1. / prodFunc.α[i])
    p_mult = prodFunc.B[i] * prodFunc.β[i] * (p / prodFunc.B[i]) ^ (1. - 1. / prodFunc.β[i])
    dsdXs = s_mult * p^(-prodFunc.θ[i])
    dsdXp = -prodFunc.θ[i] * s * p^(-prodFunc.θ[i] - 1.) .* p_mult
    return [dsdXs dxdXp; 0. p_mult]
end

function df(prodFunc::ProdFunc, Xs::Vector, Xp::Vector)
    (s, p) = f(prodFunc, Xs, Xp)
    s_mult = prodFunc.A .* prodFunc.α .* (s ./ prodFunc.A) .^ (1. - 1. / prodFunc.α)
    p_mult = prodFunc.B .* prodFunc.β .* (p ./ prodFunc.B) .^ (1. - 1. / prodFunc.β)
    dsdXs = s_mult .* p.^(-prodFunc.θ)
    dsXp = -prodFunc.θ .* s .* p.^(-prodFunc.θ - 1.) .* p_mult
    return reshape(
        vcat(dsdXs, dsXp, zeros(size(p_mult)), p_mult),
        prodFunc.n, 2, 2
    )
end


struct CSF
    w
    l
    a_w
    a_l
end

function reward(csf::CSF, i::Integer, p::Array)
    p_i = selectdim(p, ndims(p), i)
    win_proba = p_i ./ sum(p, dims = ndims(p))
    return (
        (csf.w .+ p_i .* csf.a_w) .* win_proba
        .+ (csf.l .+ p_i .* csf.a_l) .* (1. .- win_proba)
    )
end

function all_rewards(csf::CSF, p::Array)
    win_probas = p ./ sum(p, dims = ndims(p))
    return (
        (csf.w .+ p .* csf.a_w) .* win_probas
        .+ (csf.l .+ p .* csf.a_l) .* (1. .- win_probas)
    )
end

function reward_deriv(csf::CSF, i::Integer, p::Array)
    sum_ = sum(p, dims = ndims(p))
    p_i = selectdim(p, ndims(p), i)
    win_proba = p_i ./ sum_
    win_proba_deriv = (sum_ .- p_i) ./ sum_.^2
    return (
        csf.a_l .+ (csf.a_w .- csf.a_l) .* win_proba
        .+ (csf.w - csf.l .+ (csf.a_w - csf.a_l) .* p_i) .* win_proba_deriv
    )
end

function all_reward_derivs(csf::CSF, p::Array)
    sum_ = sum(p, dims = ndims(p))
    win_probas = p ./ sum_
    win_proba_derivs = (sum_ .- p) ./ sum_.^2
    return (
        csf.a_l .+ (csf.a_w - csf.a_l) .* win_probas
        + (csf.w - csf.l .+ (csf.a_w - csf.a_l) .* p) .* win_proba_derivs
    )
end


struct Problem
    n::Integer
    d::Vector
    r::Vector
    prodFunc::ProdFunc
    csf::CSF
end

Problem(d, r, prodFunc, csf) = Problem(length(d), d, r, prodFunc, csf)

function payoff(problem::Problem, i::Integer, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    σ = prod(s ./ (1. .+ s), dims = ndims(s))
    return σ .* reward(problem.csf, i, p) .- (1. .- σ) .* problem.d[i] .- problem.r[i] .* (Xs[i] + Xp[i])
end

function all_payoffs(problem::Problem, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    σ = prod(s ./ (1. .+ s), dims = ndims(s))
    return σ .* all_rewards(problem.csf, p) .- (1. .- σ) .* problem.d .- problem.r .* (Xs .+ Xp)
end

function d_all_payoffs(problem::Problem, Xs, Xp)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    probas = s ./ (1. .+ s)
    σ = prod(probas, dims = ndims(probas))
    proba_mult = σ ./ (s .* (1. .+ s))
    # prod_jac = ...
    # TO DO...
end

prodFunc = ProdFunc([10., 10.], [0.5, 0.5], [10., 10.], [0.5, 0.5], [0., 0.])
csf = CSF(1., 0., 0., 0.)
problem = Problem([1., 1.], [0.01, 0.01], prodFunc, csf)
println(payoff.(Ref(problem), 1, [1. 2.; 2. 4], [2. 4; 6. 8.]))