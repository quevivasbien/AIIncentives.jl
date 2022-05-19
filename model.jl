using Optim
using NLsolve


const DEFAULT_ITER_TOL = 1e-6
const DEFAULT_ITER_MAX_ITERS = 100
const DEFAULT_SOLVER_TOL = 1e-6
const DEFAULT_SOLVER_MAX_ITERS = 100
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
    out = prod(probas, dims = ndims(s))
    return out
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
    strats
    s
    p
    payoffs
end

function trim_to_index(result::SolverResult, index)
    return SolverResult(
        result.success,
        selectdim(result.strats, 1, index),
        selectdim(result.s, 1, index),
        selectdim(result.p, 1, index),
        selectdim(result.payoffs, 1, index)
    )
end

function prune_duplicates(result::SolverResult; atol = 1e-6, rtol=1e-1)
    if ndims(result.strats) < 3
        return result
    end
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
    return trim_to_index(result, unique)
end

function fill_from_problem(problem::Problem, result::SolverResult)
    if ndims(result.strats) > 2
        strats = reshape(result.strats, :, problem.n, 2)
        new_s = similar(strats, size(strats)[1:2])
        new_p = similar(strats, size(strats)[1:2])
        new_payoffs = similar(strats, size(strats)[1:2])
        for i in 1:size(strats)[1]
            (new_s[i, :], new_p[i, :]) = f(problem.prodFunc, strats[i, :, 1], strats[i, :, 2])
            new_payoffs[i, :] = all_payoffs_with_s_p(problem, strats[i, :, 1], strats[i, :, 2], new_s[i, :], new_p[i, :])
        end
        return SolverResult(
            result.success,
            result.strats,
            reshape(new_s, size(result.s)),
            reshape(new_p, size(result.p)),
            reshape(new_payoffs, size(result.payoffs))
        )
    else
        (new_s, new_p) = f(problem.prodFunc, result.strats[:, 1], result.strats[:, 2])
        new_payoffs = all_payoffs_with_s_p(problem, result.strats[:, 1], result.strats[:, 2], new_s, new_p)
        return SolverResult(
            result.success,
            result.strats,
            new_s,
            new_p,
            new_payoffs
        )
    end
end

function SolverResult(problem::Problem, success::Bool, strats::Array, fill = true)
    first_dims = size(strats)[1:ndims(strats)-1]
    s = similar(strats, first_dims)
    p = similar(strats, first_dims)
    payoffs = similar(strats, first_dims)
    result = SolverResult(success, strats, s, p, payoffs)
    result = prune_duplicates(result)
    if fill
        return fill_from_problem(problem, result)
    else
        return result
    end
end

function make_3d(result::SolverResult, n)
    return SolverResult(
        result.success,
        reshape(result.strats, :, n, 2),
        reshape(result.s, :, n),
        reshape(result.p, :, n),
        reshape(result.payoffs, :, n)
    )
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

function get_null_result(n)
    return SolverResult(
        false,
        fill(NaN, 1, n, 2),
        fill(NaN, 1, n),
        fill(NaN, 1, n),
        fill(NaN, 1, n)
    )
end

function resolve_multiple_solutions(
    result::SolverResult,
    problem::Problem
)
    if ndims(result.strats) == 2
        return result
    elseif size(result.strats)[1] == 1
        return trim_to_index(result, 1)
    end

    argmaxes = [x[1] for x in argmax(result.payoffs, dims=1)]
    best = argmaxes[1]
    if any(best .!= argmaxes[2:end])
        println("More than one result found; equilibrium is ambiguous")
        return get_null_result(problem)
    end
    
    return trim_to_index(result, best)
end


function solve_iters_single(problem::Problem, strat::Array)
    new_strats = similar(strat, (problem.n, 2))
    for i in 1:problem.n
        obj = get_func(problem, i, strat)
        obj_(x) = obj(exp.(x))
        jac! = get_jac(problem, i, strat, inplace = true)
        jac_!(var, x) = jac!(var, exp.(x))
        # lower_bound = [0., 0.]
        # upper_bound = [Inf, Inf]
        init_guess = strat[i, :]
        res = optimize(
            obj_, jac_!,
            # lower_bound, upper_bound,
            log.(init_guess),
            NewtonTrustRegion(),
            Optim.Options(
                x_tol = DEFAULT_SOLVER_TOL,
                iterations = DEFAULT_SOLVER_MAX_ITERS
            )
        )
        if any(problem.prodFunc.θ .> 0.)
            # check the zero-input payoff
            x = copy(strat)
            x[i, :] = [0., 0.]
            (_, p) = f(problem.prodFunc, x[:, 1], x[:, 2])
            zero_payoff = reward(problem.csf, i, p)
            if zero_payoff > -Optim.minimum(res)
                new_strats[i, :] = [0., 0.]
            else
                new_strats[i, :] = exp.(Optim.minimizer(res))
            end
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
            return SolverResult(problem, true, new_strat)
        end
        strat = new_strat
    end
    if verbose
        println("Reached max iterations")
    end
    return SolverResult(problem, false, strat)
end

function solve_iters(
    problem::Problem;
    init_guess::Number = 1.,
    max_iters = DEFAULT_ITER_MAX_ITERS,
    tol = DEFAULT_ITER_TOL,
    verbose = true
)
    return solve_iters(problem, fill(init_guess, (problem.n, 2)); max_iters, tol, verbose)
end


function solve_roots(
    problem::Problem;
    init_guesses::Vector{Float64} = [10.0^i for i in -5:5],
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
    # for i in 1:n_guesses
    #     println("$i is success: ", successes[i])
    #     println("$i result: ", results[i, :, :])
    # end
    if !any(successes)
        println("Roots solver failed to converge from the given initial guesses!")
        return get_null_result(problem)
    end
    results = results[successes, :, :]
    solverResult = SolverResult(problem, true, results)
    println("Shape of roots solution: ", size(solverResult.strats))
    if resolve_multiple
        return resolve_multiple_solutions(solverResult, problem)
    else
        return solverResult
    end
end


function solve_hybrid(
    problem::Problem;
    init_guesses::Vector{Float64} = [10.0^i for i in -5:5],
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
    strats = reshape(roots_sol.strats, :, problem.n, 2)  # just for if there's only one solution
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
        return get_null_result(problem)
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
    println("With `solve_iters`: ", solve_iters_sol)
    println("With `solve_roots`: ", solve_roots_sol)
    println("With `solve_hybrid`: ", solve_hybrid_sol)
end
    
    
if abspath(PROGRAM_FILE) == @__FILE__
    test()
end
