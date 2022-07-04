include("./ProdFunc.jl")


function get_total_safety(s::AbstractArray)
    probas = s ./ (1. .+ s)
    # if s is infinite, proba should be 1
    probas[isnan.(s)] .= 1.
    return prod(probas, dims = ndims(s))
end

function get_total_safety(s::AbstractVector)
    probas = s ./ (1. .+ s)
    probas[isnan.(s)] .= 1.
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



struct SolverResult
    success::Bool
    Xs::Array
    Xp::Array
    s::Array
    p::Array
    payoffs::Array
end

function trim_to_index(result::SolverResult, index)
    return SolverResult(
        result.success,
        copy(selectdim(result.Xs, 1, index)),
        copy(selectdim(result.Xp, 1, index)),
        copy(selectdim(result.s, 1, index)),
        copy(selectdim(result.p, 1, index)),
        copy(selectdim(result.payoffs, 1, index))
    )
end

function prune_duplicates(result::SolverResult; atol = 1e-6, rtol=5e-2)
    if ndims(result.Xs) < 2
        return result
    end
    dups = Vector{Integer}()
    unique = Vector{Integer}()
    n_results = size(result.Xs, 1)
    for i in 1:n_results
        if i ∈ dups
            continue
        end
        strats1 =  hcat(selectdim(result.Xs, 1, i), selectdim(result.Xp, 1, i))
        for j in (i+1):n_results
            strats2 = hcat(selectdim(result.Xs, 1, j), selectdim(result.Xp, 1, j))
            if isapprox(strats1, strats2; atol = atol, rtol = rtol)
                push!(dups, j)
            end
        end
        push!(unique, i)
    end
    return trim_to_index(result, unique)
end

function get_s_p_payoffs(problem::Problem, Xs_, Xp_)
    if ndims(Xs_) > 1
        Xs = reshape(Xs_, :, problem.n)
        Xp = reshape(Xp_, :, problem.n)
        s = similar(Xs)
        p = similar(Xs)
        payoffs = similar(Xs)
        for i in 1:size(Xs)[1]
            (s[i, :], p[i, :]) = f(problem.prodFunc, Xs[i, :], Xp[i, :])
            payoffs[i, :] = all_payoffs_with_s_p(problem, Xs[i, :], Xp[i, :], s[i, :], p[i, :])
        end
        return (
            reshape(s, size(Xs_)),
            reshape(p, size(Xs_)),
            reshape(payoffs, size(Xs_))
        )
    else
        (s, p) = f(problem.prodFunc, Xs_, Xp_)
        payoffs = all_payoffs_with_s_p(problem, Xs_, Xp_, s, p)
        return s, p, payoffs
    end
end

function SolverResult(problem::Problem, success::Bool, Xs, Xp; fill = true, prune = true)
    result = if fill
        SolverResult(success, Xs, Xp, get_s_p_payoffs(problem, Xs, Xp)...)
    else
        SolverResult(success, Xs, Xp, similar(Xs), similar(Xs), similar(Xs))
    end
    if prune
        return prune_duplicates(result)
    else
        return result
    end
end

function make_3d(result::SolverResult, n)
    return SolverResult(
        result.success,
        reshape(result.Xs, :, n),
        reshape(result.Xp, :, n),
        reshape(result.s, :, n),
        reshape(result.p, :, n),
        reshape(result.payoffs, :, n)
    )
end

function Base.:+(result1::SolverResult, result2::SolverResult)
    return SolverResult(
        result1.success && result2.success,
        cat(result1.Xs, result2.Xs, dims = 1),
        cat(result1.Xp, result2.Xp, dims = 1),
        cat(result1.s, result2.s, dims = 1),
        cat(result1.p, result2.p, dims = 1),
        cat(result1.payoffs, result2.payoffs, dims = 1)
    )
end

function get_null_result(n)
    return SolverResult(
        false,
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n)
    )
end

function resolve_multiple_solutions(
    result::SolverResult,
    problem::Problem
)
    if ndims(result.Xs) == 1
        return result
    elseif size(result.Xs)[1] == 1
        return trim_to_index(result, 1)
    end

    argmaxes = [x[1] for x in argmax(result.payoffs, dims=1)]
    best = argmaxes[1]
    if any(best .!= argmaxes[2:end])
        println("More than one result found; equilibrium is ambiguous")
        return get_null_result(problem.n)
    end
    
    return trim_to_index(result, best)
end

function print(result::SolverResult)
    println("success: ", result.success)
    println("Xs: ", result.Xs)
    println("Xp: ", result.Xp)
    println("s: ", result.s)
    println("p: ", result.p)
    println("payoffs: ", result.payoffs, '\n')
end
