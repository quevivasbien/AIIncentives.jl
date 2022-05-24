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


struct CSF
    w::Number
    l::Number
    a_w::Number
    a_l::Number
end

function reward(csf::CSF, i::Integer, p::Vector)
    sum_ = sum(p)
    if sum_ == 0.
        return 1. / length(p)
    end
    win_proba = p[i] / sum_
    return (
        (csf.w + p[i] * csf.a_w) * win_proba
        + (csf.l + p[i] * csf.a_l) * (1. - win_proba)
    )
end

function all_rewards(csf::CSF, p::Vector)
    sum_ = sum(p)
    if sum_ == 0.
        return fill(1. / length(p), length(p))
    end
    win_probas = p ./ sum_
    return (
        (csf.w .+ p .* csf.a_w) .* win_probas
        .+ (csf.l .+ p .* csf.a_l) .* (1. .- win_probas)
    )
end

function reward_deriv(csf::CSF, i::Integer, p::Vector)
    sum_ = sum(p)
    if sum_ == 0.
        return Inf
    end
    win_proba = p[i] / sum_
    win_proba_deriv = (sum_ .- p[i]) ./ sum_.^2
    return (
        csf.a_l + (csf.a_w - csf.a_l) * win_proba
        + (csf.w - csf.l + (csf.a_w - csf.a_l) * p[i]) * win_proba_deriv
    )
end

function all_reward_derivs(csf::CSF, p::Vector)
    sum_ = sum(p)
    if sum_ == 0.
        return fill(Inf, length(p))
    end
    win_probas = p ./ sum_
    win_proba_derivs = (sum_ .- p) ./ sum_.^2
    return (
        csf.a_l .+ (csf.a_w - csf.a_l) .* win_probas
        + (csf.w - csf.l .+ (csf.a_w - csf.a_l) .* p) .* win_proba_derivs
    )
end

function reward_and_deriv(csf::CSF, i::Integer, p::Vector)
    sum_ = sum(p)
    if sum_ == 0.
        return (1. / length(p), Inf)
    end
    win_proba = p[i] ./ sum_
    win_proba_deriv = (sum_ .- p[i]) ./ sum_.^2
    reward = (csf.w + p[i] * csf.a_w) * win_proba + (csf.l + p[i] * csf.a_l) * (1. - win_proba)
    return (
        reward,
        csf.a_l + (csf.a_w - csf.a_l) * win_proba
        + (csf.w - csf.l + (csf.a_w - csf.a_l) * p[i]) * win_proba_deriv
    )
end

function all_rewards_and_derivs(csf::CSF, p::Vector)
    sum_ = sum(p)
    if sum_ == 0.
        return (fill(1 / length(p), length(p)), fill(Inf, length(p)))
    end
    win_probas = p ./ sum_
    win_proba_derivs = (sum_ .- p) ./ sum_.^2
    rewards = (csf.w .+ p .* csf.a_w) .* win_probas .+ (csf.l .+ p .* csf.a_l) .* (1. .- win_probas)
    return (
        rewards,
        csf.a_l .+ (csf.a_w - csf.a_l) .* win_probas
        + (csf.w - csf.l .+ (csf.a_w - csf.a_l) .* p) .* win_proba_derivs
    )
end
