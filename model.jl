using Optim
using NLsolve


const DEFAULT_ITER_TOL = 1e-4
const DEFAULT_ITER_MAX_ITERS = 100
const DEFAULT_SOLVER_TOL = 1e-5
const DEFAULT_SOLVER_MAX_ITERS = 400
const EPSILON = 1e-8

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

function get_total_safety(s::Array)
    probas = s ./ (1. .+ s)
    # if s is infinite, proba should be 1
    probas[isnan.(probas)] .= 1.
    return prod(probas, dims = ndims(s))
end

function get_total_safety(s::Vector)
    probas = s ./ (1. .+ s)
    probas[isnan.(probas)] .= 1
    return prod(probas)
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
        # win_proba = 1 / n
        return (csf.w + csf.l) / length(p)
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
        # win_probas = [1/n, ..., 1/n]
        return fill((csf.w + csf.l) / length(p), length(p))
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
        return ((csf.w + csf.l) / length(p), Inf)
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
        return (fill((csf.w + csf.l) / length(p), length(p)), fill(Inf, length(p)))
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



struct SolverResult{T <: AbstractArray}
    success::Bool
    Xs::T
    Xp::T
    s::T
    p::T
    payoffs::T
end

function trim_to_index(result::SolverResult, index)
    return SolverResult(
        result.success,
        selectdim(result.Xs, 1, index),
        selectdim(result.Xp, 1, index),
        selectdim(result.s, 1, index),
        selectdim(result.p, 1, index),
        selectdim(result.payoffs, 1, index)
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
        fill(NaN, 1, n),
        fill(NaN, 1, n),
        fill(NaN, 1, n),
        fill(NaN, 1, n),
        fill(NaN, 1, n)
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
    println("Xs: ", result.Xs)
    println("Xp: ", result.Xp)
    println("s: ", result.s)
    println("p: ", result.p)
    println("payoffs: ", result.payoffs, '\n')
end


function solve_iters_single(problem::Problem, strat::Array)
    new_strats = similar(strat)
    for i in 1:problem.n
        obj = get_func(problem, i, strat)
        obj_(x) = obj(exp.(x))
        jac! = get_jac(problem, i, strat, inplace = true)
        jac_!(var, x) = jac!(var, exp.(x))
        # lower_bound = [0., 0.]
        # upper_bound = [Inf, Inf]
        init_guess = strat[i, :]
        # if init_guess is zero-effort, try breaking out
        if all(init_guess .== 0.)
            init_guess .= EPSILON
        end
        res = optimize(
            obj_, jac_!,
            # lower_bound, upper_bound,
            log.(init_guess),
            NewtonTrustRegion(initial_delta = 0.01, delta_hat = 0.1),
            Optim.Options(
                x_tol = DEFAULT_SOLVER_TOL,
                iterations = DEFAULT_SOLVER_MAX_ITERS
            )
        )
        # check the zero-input payoff
        zero_payoff = if any(problem.prodFunc.θ .> 0.)
            x = copy(strat)
            x[i, :] = [0., 0.]
            (_, p) = f(problem.prodFunc, x[:, 1], x[:, 2])
            # safety in this case is ∞
            reward(problem.csf, i, p)
        else
            # safety is 0
            -problem.d[i]
        end
        # zero_payoff = begin
        #     x = copy(strat)
        #     x[i, :] .= EPSILON
        #     payoff(problem, i, x[:, 1], x[:, 2])
        # end
        if zero_payoff > -Optim.minimum(res)
            new_strats[i, :] = [0., 0.]
        else
            new_strats[i, :] = exp.(Optim.minimizer(res))
        end
    end
    return new_strats
end
    
function solve_iters(
    problem::Problem,
    init_guess::Array;
    max_iters = DEFAULT_ITER_MAX_ITERS,
    tol = DEFAULT_ITER_TOL,
    verbose = true
)
    strat = init_guess
    for t in 1:max_iters
        new_strat = solve_iters_single(problem, strat)
        if maximum(abs.(new_strat - strat) ./ (strat .+ EPSILON)) < tol
            if verbose
                println("Exited on iteration ", t)
            end
            return SolverResult(problem, true, new_strat[:, 1], new_strat[:, 2], prune = false)
        end
        strat = new_strat
    end
    if verbose
        println("Reached max iterations")
    end
    return SolverResult(problem, false, strat[:, 1], strat[:, 2])
end

function solve_iters(
    problem::Problem;
    init_guess::Number = 1.,
    max_iters = DEFAULT_ITER_MAX_ITERS,
    tol = DEFAULT_ITER_TOL,
    verbose = true
)
    # return solve_iters(problem, exp.(randn(problem.n, 2)); max_iters, tol, verbose)
    return solve_iters(problem, fill(init_guess, (problem.n, 2)); max_iters, tol, verbose)
end


function solve_roots(
    problem::Problem;
    init_guesses::Vector{Float64} = [10.0^(3*i) for i in -2:2],
    resolve_multiple = true,
    ftol = 1e-8
)
    jac! = get_jac(problem, inplace = true)
    function obj!(val, x)
        # reshape x to block format as expected by jac! function
        # also take exponential to force positive bounds
        y = reshape(exp.(x), (problem.n, 2))
        # fill jacobian at y
        jac!(val, y)
    end
    n_guesses = length(init_guesses)
    results = Array{Float64}(undef, n_guesses, problem.n, 2)
    successes = falses(n_guesses)
    Threads.@threads for i in 1:n_guesses
        init_guess = fill(init_guesses[i], problem.n * 2)
        res = nlsolve(
            obj!,
            log.(init_guess),
            ftol = ftol,
            method = :trust_region  # this is just the default
        )
        if res.f_converged
            successes[i] = true
            results[i, :, :] = reshape(exp.(res.zero), (problem.n, 2))
        end
    end
    if !any(successes)
        println("Roots solver failed to converge from the given initial guesses!")
        return get_null_result(problem.n)
    end
    results = results[successes, :, :]
    solverResult = SolverResult(problem, true, results[:, :, 1], results[:, :, 2])
    if resolve_multiple
        return resolve_multiple_solutions(solverResult, problem)
    else
        return solverResult
    end
end


function solve_hybrid(
    problem::Problem;
    init_guesses::Vector{Float64} = [10.0^(3*i) for i in -2:2],
    max_iters = DEFAULT_ITER_MAX_ITERS, tol = DEFAULT_ITER_TOL,
    verbose = false
)
    if verbose
        println("Finding roots...")
    end
    roots_sol = solve_roots(problem, init_guesses = init_guesses, ftol = 1e-4, resolve_multiple = false)
    if !roots_sol.success
        return roots_sol
    end
    strats = cat(
        # reshape needed if there's only 1 solution
        reshape(roots_sol.Xs, :, problem.n),
        reshape(roots_sol.Xp, :, problem.n),
        dims = 3
    )
    n_tries = size(strats)[1]
    converged = falses(n_tries)
    results = Vector{SolverResult}(undef, n_tries)
    if verbose
        println("Iterating...")
    end
    Threads.@threads for i in 1:n_tries
        strats_ = copy(selectdim(strats, 1, i))
        iter_sol = solve_iters(problem, strats_; max_iters, tol, verbose)
        if iter_sol.success
            converged[i] = true
            results[i] = iter_sol
        end
    end
    if !any(converged)
        return get_null_result(problem.n)
    end
    combined_sols = sum([make_3d(r, problem.n) for (r, c) in zip(results, converged) if c])
    return resolve_multiple_solutions(prune_duplicates(combined_sols), problem)
end
    
    
function test()
    prodFunc = ProdFunc([10., 10.], [0.5, 0.5], [10., 10.], [0.5, 0.5], [0., 0.])
    csf = CSF(1., 0., 0., 0.)
    problem = Problem([1., 1.], [0.01, 0.01], prodFunc, csf)
    @time solve_iters_sol = solve_iters(problem)
    @time solve_roots_sol = solve_roots(problem)
    @time solve_hybrid_sol = solve_hybrid(problem)
    println("With `solve_iters`:")
    print(solve_iters_sol)
    println("With `solve_roots`:")
    print(solve_roots_sol)
    println("With `solve_hybrid`:")
    print(solve_roots_sol)
    return
end
