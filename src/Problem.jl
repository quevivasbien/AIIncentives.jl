function get_total_safety(s::AbstractArray)
    probas = s ./ (1. .+ s)
    # if s is infinite, proba should be 1
    probas[isnan.(s) .| isinf.(s)] .= 1.
    return prod(probas, dims = ndims(s))
end

function get_total_safety(s::AbstractVector)
    probas = s ./ (1. .+ s)
    probas[isnan.(s) .| isinf.(s)] .= 1.
    return prod(probas)
end


struct Problem
    n::Integer
    d::Vector
    r::Vector
    prodFunc::ProdFunc
    csf::CSF
end

Problem(d, r, prodFunc, csf) = Problem(length(d), d, r, prodFunc, csf)

Problem(
    ;
    d = [0., 0.],
    r = [0.1, 0.1],
    prodFunc = ProdFunc(),
    csf = CSF()
) = Problem(2, d, r, prodFunc, csf)

function payoff(problem::Problem, i::Integer, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    σ = get_total_safety(s)
    return σ .* reward(problem.csf, i, p) .- (1. .- σ) .* problem.d[i] .- problem.r[i] .* (Xs[i] + Xp[i])
end

function payoff_deriv(problem::Problem, i::Integer, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    σ = get_total_safety(s)
    proba_mult = σ / (s[i] * (1. + s[i]))
    prod_jac = df_from_s_p(problem.prodFunc, i, s[i], p[i])  # a 2 x 2 array
    s_ks = prod_jac[1, 1]
    s_kp = prod_jac[1, 2]
    p_ks = prod_jac[2, 1]
    p_kp = prod_jac[2, 2]
    σ_ks = proba_mult * s_ks
    σ_kp = proba_mult * s_kp
    (q, q_p) = reward_and_deriv(problem.csf, i, p)
    payoffs_Xs = σ_ks * (q + problem.d[i]) + σ * q_p * p_ks - problem.r[i]
    payoffs_Xp = σ_kp * (q + problem.d[i]) + σ * q_p * p_kp - problem.r[i]
    return [payoffs_Xs, payoffs_Xp]
end

function all_payoffs_with_s_p(problem::Problem, Xs::Vector, Xp::Vector, s::Vector, p::Vector)
    σ = get_total_safety(s)
    return σ .* all_rewards(problem.csf, p) .- (1. .- σ) .* problem.d .- problem.r .* (Xs .+ Xp)
end

function all_payoffs(problem::Problem, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    return all_payoffs_with_s_p(problem, Xs, Xp, s, p)
end

function all_payoffs_deriv_flat(problem::Problem, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    σ = get_total_safety(s)
    proba_mult = σ ./ (s .* (1. .+ s))
    prod_jac = df(problem.prodFunc, Xs, Xp)  # an n x 2 x 2 array
    s_ks = prod_jac[:, 1, 1]
    s_kp = prod_jac[:, 1, 2]
    p_ks = prod_jac[:, 2, 1]
    p_kp = prod_jac[:, 2, 2]
    σ_ks = proba_mult .* s_ks
    σ_kp = proba_mult .* s_kp
    (q, q_p) = all_rewards_and_derivs(problem.csf, p)
    payoffs_Xs = σ_ks .* (q .+ problem.d) .+ σ .* q_p .* p_ks .- problem.r
    payoffs_Xp = σ_kp .* (q .+ problem.d) .+ σ .* q_p .* p_kp .- problem.r
    return vcat(payoffs_Xs, payoffs_Xp)
end

function all_payoffs_deriv(problem::Problem, Xs::Vector, Xp::Vector)
    return reshape(all_payoffs_deriv_flat(problem, Xs, Xp), (self.n, 2, 2))
end

function get_func(problem::Problem, i::Integer, strats::Array)
    strats_ = copy(strats)
    function func(x)
        strats_[i, :] = x
        return -payoff(problem, i, strats_[:, 1], strats_[:, 2])
    end
end

function get_jac(problem::Problem, i::Integer, strats::Array; inplace = false)
    # returns gradient of player i's payoff given other players' strats
    strats_ = copy(strats)
    if inplace
        function jac!(grad, x)
            strats_[i, :] = x
            copy!(grad, -payoff_deriv(problem, i, strats_[:, 1], strats_[:, 2]))
        end
        return jac!
    else
        function jac(x)
            strats_[i, :] = x
            return -payoff_deriv(problem, i, strats_[:, 1], strats_[:, 2])
        end
        return jac
    end
end

function get_jac(problem::Problem; inplace = false)
    # returns flat jacobian of all players' payoffs
    if inplace
        function jac!(grad, x)
            copy!(grad, -all_payoffs_deriv_flat(problem, x[:, 1], x[:, 2]))
        end
        return jac!
    else
        function jac(x)
            return -all_payoffs_deriv_flat(problem, x[:, 1], x[:, 2])
        end
        return jac
    end
end

function is_symmetric(problem::Problem)
    (
        is_symmetric(problem.prodFunc)
        && all(problem.d[1] .== problem.d[2:problem.n])
        && all(problem.r[1] .== problem.r[2:problem.n])
    )
end
