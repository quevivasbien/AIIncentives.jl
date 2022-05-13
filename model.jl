using Optim
using NLsolve


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
    win_proba = p[i] / sum(p)
    return (
        (csf.w .+ p[i] .* csf.a_w) .* win_proba
        .+ (csf.l .+ p[i] .* csf.a_l) .* (1. .- win_proba)
    )
end

function all_rewards(csf::CSF, p::Vector)
    win_probas = p ./ sum(p)
    return (
        (csf.w .+ p .* csf.a_w) .* win_probas
        .+ (csf.l .+ p .* csf.a_l) .* (1. .- win_probas)
    )
end

function reward_deriv(csf::CSF, i::Integer, p::Vector)
    sum_ = sum(p)
    win_proba = p[i] / sum_
    win_proba_deriv = (sum_ .- p[i]) ./ sum_.^2
    return (
        csf.a_l + (csf.a_w - csf.a_l) * win_proba
        + (csf.w - csf.l + (csf.a_w - csf.a_l) * p[i]) * win_proba_deriv
    )
end

function all_reward_derivs(csf::CSF, p::Vector)
    sum_ = sum(p)
    win_probas = p ./ sum_
    win_proba_derivs = (sum_ .- p) ./ sum_.^2
    return (
        csf.a_l .+ (csf.a_w - csf.a_l) .* win_probas
        + (csf.w - csf.l .+ (csf.a_w - csf.a_l) .* p) .* win_proba_derivs
    )
end

function reward_and_deriv(csf::CSF, i::Integer, p::Vector)
    sum_ = sum(p)
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
    win_probas = p ./ sum_
    win_proba_derivs = (sum_ .- p) ./ sum_.^2
    rewards = (csf.w .+ p .* csf.a_w) .* win_probas .+ (csf.l .+ p .* csf.a_l) .* (1. .- win_probas)
    return (
        rewards,
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
    σ = prod(s ./ (1. .+ s))
    return σ .* reward(problem.csf, i, p) .- (1. .- σ) .* problem.d[i] .- problem.r[i] .* (Xs[i] + Xp[i])
end

function payoff_deriv(problem::Problem, i::Integer, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    σ = prod(s ./ (1. .+ s))
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
    σ = prod(s ./ (1. .+ s))
    return σ .* all_rewards(problem.csf, p) .- (1. .- σ) .* problem.d .- problem.r .* (Xs .+ Xp)
end

function all_payoffs(problem::Problem, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    return all_payoffs_with_s_p(problem, Xs, Xp, s, p)
end

function all_payoffs_deriv_flat(problem::Problem, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    σ = prod(s ./ (1. .+ s))
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

function get_jac(problem::Problem, i::Integer, strats::Array)
    # returns gradient of player i's payoff given other players' strats
    strats_ = copy(strats)
    function jac!(grad, x)
        strats_[i, :] = x
        copy!(grad, -payoff_deriv(problem, i, strats_[:, 1], strats_[:, 2]))
    end
    return jac!
end

function get_jac(problem::Problem)
    # returns flat jacobian of all players' payoffs
    function jac!(grad, x)
        copy!(grad, -all_payoffs_deriv_flat(problem, x[:, 1], x[:, 2]))
    end
    return jac!
end




mutable struct SolverResult
    success::Bool
    strats
    s
    p
    payoffs
end

function trim_to_index!(result::SolverResult, index)
    result.strats = selectdim(result.strats, 1, index)
    result.s = selectdim(result.s, 1, index)
    result.p = selectdim(result.p, 1, index)
    result.payoffs = selectdim(result.payoffs, 1, index)
end

function prune_duplicates!(result::SolverResult, atol = 1e-6, rtol=1e-1)
    dups = Vector{Integer}()
    unique = Vector{Integer}()
    n_results = size(result.strats, 1)
    for i in 1:n_results
        if i ∈ dups
            continue
        end
        strats1 = selectdim(result.strats, 1, i)
        for j in (i+1):n_results
            strats2 = selectdim(result.strats, 1, j)
            if isapprox(strats1, strats2; atol = atol, rtol = rtol)
                push!(dups, j)
            end
        end
        push!(unique, i)
    end
    trim_to_index!(result, unique)
end

function fill_from_problem!(problem::Problem, result::SolverResult)
    if ndims(result.strats) > 2
        strats = reshape(result.strats, :, problem.n, 2)
        new_s = similar(strats, size(strats)[1:2])
        new_p = similar(strats, size(strats)[1:2])
        new_payoffs = similar(strats, size(strats)[1:2])
        for i in 1:size(strats)[1]
            (new_s[i, :], new_p[i, :]) = f(problem.prodFunc, strats[i, :, 1], strats[i, :, 2])
            new_payoffs[i, :] = all_payoffs_with_s_p(problem, strats[i, :, 1], strats[i, :, 2], new_s[i, :], new_p[i, :])
        end
        copyto!(result.s, new_s)
        copyto!(result.p, new_p)
        copyto!(result.payoffs, new_payoffs)
    else
        (result.s, result.p) = f(problem.prodFunc, result.strats[:, 1], result.strats[:, 2])
        result.payoffs = all_payoffs_with_s_p(problem, result.strats[:, 1], result.strats[:, 2], result.s, result.p)
    end
end

function SolverResult(problem::Problem, success::Bool, strats::Array, fill = true)
    first_dims = size(strats)[1:ndims(strats)-1]
    s = similar(strats, first_dims)
    p = similar(strats, first_dims)
    payoffs = similar(strats, first_dims)
    result = SolverResult(success, strats, s, p, payoffs)
    prune_duplicates!(result)
    if fill
        fill_from_problem!(problem, result)
    end
    return result
end

function Base.:+(result1::SolverResult, result2::SolverResult)
    return SolverResult(
        result1.success && result2.success,
        cat(result1.strats, result2.strats, dims = 1),
        cat(result1.s, result2.s, dims = 1),
        cat(result1.p, result2.p, dims = 1),
        cat(result1.payoffs, result2.payoffs, dims = 1)
    )
end


function get_null_result(problem::Problem)
    return SolverResult(
        false,
        fill(NaN, problem.n),
        fill(NaN, problem.n),
        fill(NaN, problem.n),
        fill(NaN, problem.n)
    )
end
        
function _solve_iters_single(problem::Problem, strat::Array)
    new_strats = similar(strat, (problem.n, 2))
    for i in 1:problem.n
        obj = get_func(problem, i, strat)
        jac! = get_jac(problem, i, strat)
        lower_bound = [0., 0.]
        upper_bound = [Inf, Inf]
        init_guess = strat[i, :]
        res = optimize(
            obj, jac!, lower_bound, upper_bound, init_guess, Fminbox(BFGS())
        )
        new_strats[i, :] = Optim.minimizer(res)
    end
    return new_strats
end
    
function solve_iters(problem::Problem, init_guess::Number = 1., max_iters::Integer = 100, iter_tol = 1e-8)
    strat = fill(init_guess, (problem.n, 2))
    for t in 1:max_iters
        new_strat = _solve_iters_single(problem, strat)
        if maximum(abs.(new_strat - strat) ./ strat) < iter_tol
            println("Exited on iteration ", t)
            return SolverResult(problem, true, new_strat)
        end
        strat = new_strat
    end
    println("Reached max iterations")
    return SolverResult(problem, false, strat)
end


function solve_roots(
    problem::Problem,
    init_guesses::Vector{Float64} = [10.0^i for i in -5:5],
    # max_iters::Integer = 100, tol = 1e-8
)
    jac! = get_jac(problem)
    function obj!(val, x)
        # reshape x and transform to positive value
        y = exp.(reshape(x, (problem.n, 2)))
        # fill jacobian at y
        jac!(val, y)
    end
    n_guesses = length(init_guesses)
    results = Array{Float64}(undef, n_guesses, problem.n, 2)
    successes = fill(false, n_guesses)
    for i in 1:n_guesses
        init_guess = fill(log(init_guesses[i]), problem.n * 2)
        res = nlsolve(
            obj!,
            init_guess
        )
        if res.f_converged
            successes[i] = true
            results[i, :, :] = exp.(reshape(res.zero, (problem.n, 2)))
        end
    end
    if !any(successes)
        println("Roots solver failed to converge from the given initial guesses!")
        return get_null_result(problem)
    end
    results = results[successes, :, :]
    return SolverResult(problem, true, results)
end
    
    
function test()
    prodFunc = ProdFunc([10., 10.], [0.5, 0.5], [10., 10.], [0.5, 0.5], [0., 0.])
    csf = CSF(1., 0., 0., 0.)
    problem = Problem([1., 1.], [0.01, 0.01], prodFunc, csf)
    # println(df(prodFunc, [1., 1.], [2., 2.]))
    @time println("With `solve_iters`: ", solve_iters(problem))
    @time println("With `solve_hybrid`: ", solve_hybrid(problem))
end
    
    
if abspath(PROGRAM_FILE) == @__FILE__
    test()
end