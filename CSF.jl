struct CSF
    w::Number
    l::Number
    a_w::Number
    a_l::Number
end

CSF(; w = 1., l = 0., a_w = 0., a_l = 0.) = CSF(w, l, a_w, a_l)

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

(csf::CSF)(i::Integer, p::Vector) = reward(csf, i, p)

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

(csf::CSF)(p::Vector) = all_rewards(csf, p)

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
