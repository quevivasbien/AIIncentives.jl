using Optim
using NLsolve

include("./Problem.jl")

struct IterOptions
    tol::Number
    max_iters::Integer
    solver_tol::Number
    solver_max_iters::Integer
    trust_delta_init::Number
    trust_delta_max::Number
    init_guess::Number
    n_init_points::Integer
    init_mu::Number
    init_sigma::Number
    verbose::Bool
end

IterOptions(
    ;
    tol::Number = 1e-6, max_iters::Integer = 1000,
    solver_tol::Number = 1e-8, solver_max_iters::Integer = 500,
    trust_delta_init::Number = 0.01, trust_delta_max::Number = 0.1,
    init_guess::Number = 1., n_init_points::Integer = 20,
    init_mu::Number = 0., init_sigma::Number = 1.,
    verbose::Bool = false
) = IterOptions(
    tol, max_iters,
    solver_tol, solver_max_iters,
    trust_delta_init, trust_delta_max,
    init_guess, n_init_points,
    init_mu, init_sigma,
    verbose
)

const DEFAULT_OPTIONS = IterOptions()

# ITERATING METHOD `solve_iters`

function single_iter_for_i(
    problem, strat, i, init_guess,
    options = DEFAULT_OPTIONS
)
    obj = get_func(problem, i, strat)
    obj_(x) = obj(exp.(x))
    jac! = get_jac(problem, i, strat, inplace = true)
    jac_!(var, x) = jac!(var, exp.(x))
    # if init_guess is zero-effort, try breaking out
    if all(init_guess .== 0.)
        init_guess = exp.(options.init_mu .+ options.init_sigma .* randn(2))
    end
    res = optimize(
        obj_, jac_!,
        log.(init_guess),
        # NewtonTrustRegion(initial_delta = options.trust_delta_init, delta_hat = options.trust_delta_max),
        LBFGS(),
        Optim.Options(
            x_tol = options.solver_tol,
            iterations = options.solver_max_iters
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
    if zero_payoff > -Optim.minimum(res)
        return [0., 0.]
    else
        return exp.(Optim.minimizer(res))
    end
end

function solve_iters_single(
    problem::Problem, strat::Array,
    options = DEFAULT_OPTIONS
)
    new_strats = similar(strat)
    for i in 1:problem.n
        new_strats[i, :] = single_iter_for_i(problem, strat, i, strat[i, :], options)
    end
    return new_strats
end
    
function solve_iters(
    problem::Problem,
    init_guess::Array;
    options = DEFAULT_OPTIONS
)
    strat = init_guess
    for t in 1:options.max_iters
        new_strat = solve_iters_single(problem, strat, options)
        if maximum(abs.(new_strat - strat) ./ (strat .+ EPSILON)) < options.tol
            if options.verbose
                println("Exited on iteration $t")
            end
            return SolverResult(problem, true, new_strat[:, 1], new_strat[:, 2], prune = false)
        end
        strat = new_strat
    end
    if options.verbose
        println("Reached max iterations")
    end
    return SolverResult(problem, false, strat[:, 1], strat[:, 2])
end

function solve_iters(
    problem::Problem;
    options = DEFAULT_OPTIONS
)
    return solve_iters(problem, fill(options.init_guess, (problem.n, 2)); options)
end


# SCATTER ITERATING METHOD:
# Runs iter_solve from multiple init points and compares results

function solve_scatter(
    problem::Problem;
    options = DEFAULT_OPTIONS
)
    # draw init points from log-normal distribution
    init_points = exp.(options.init_mu .+ options.init_sigma .* randn(options.n_init_points, problem.n, 2))
    results = SolverResult[]
    Threads.@threads for i in 1:options.n_init_points
        result = solve_iters(
            problem,
            init_points[i, :, :];
            options
        )
        if result.success
            push!(results, result)
        end
    end
    if length(results) == 0
        println("None of the solver iterations converged.")
        return get_null_result(problem.n)
    end
    combined_sols = sum([make_3d(r, problem.n) for r in results])
    return combined_sols
end


# ROOT-FINDING METHOD `solve_roots`

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


# HYBRID METHOD (Runs root-finding method, then iterating method)

function solve_hybrid(
    problem::Problem;
    init_guesses::Vector{Float64} = [10.0^(3*i) for i in -2:2],
    options = DEFAULT_OPTIONS
)
    if options.verbose
        println("Finding roots...")
    end
    roots_sol = solve_roots(problem, init_guesses = init_guesses, ftol = options.solver_tol, resolve_multiple = false)
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
    results = SolverResult[]
    if options.verbose
        println("Iterating...")
    end
    Threads.@threads for i in 1:n_tries
        strats_ = copy(selectdim(strats, 1, i))
        iter_sol = solve_iters(problem, strats_; options)
        if iter_sol.success
            push!(results, iter_sol)
        end
    end
    if length(results) == 0
        return get_null_result(problem.n)
    end
    combined_sols = sum([make_3d(r, problem.n) for r in results])
    return resolve_multiple_solutions(prune_duplicates(combined_sols), problem)
end


# Solver for mixed strategy equilibria
# Runs iterating solver over history of run

function single_mixed_iter_for_i(problem, history, i, init_guess, options = DEFAULT_OPTIONS)
    objs = [get_func(problem, i, history[:, :, j]) for j in size(history, 3)]
    obj_(x) = sum(obj(exp.(x)) for obj in objs)
    jacs = [get_jac(problem, i, history[:, :, j], inplace = false) for j in size(history, 3)]
    jac_!(var, x) = copy!(var, sum(jac(exp.(x)) for jac in jacs))
    # if init_guess is zero-effort, try breaking out
    if all(init_guess .== 0.)
        init_guess = exp.(options.init_mu .+ options.init_sigma .* randn(2))
    end
    res = optimize(
        obj_, jac_!,
        log.(init_guess),
        # NewtonTrustRegion(initial_delta = options.trust_delta_init, delta_hat = options.trust_delta_max),
        LBFGS(),
        Optim.Options(
            x_tol = options.solver_tol,
            iterations = options.solver_max_iters
        )
    )
    # check the zero-input payoff
    zero_payoff = obj_([0., 0.])
    if zero_payoff > -Optim.minimum(res)
        return [0., 0.]
    else
        return exp.(Optim.minimizer(res))
    end
end

function single_mixed_iter(problem, history, init_guess, options = DEFAULT_OPTIONS)
    new_strat = Array{Float64}(undef, problem.n, 2)
    for i in 1:problem.n
        new_strat[i, :] = single_mixed_iter_for_i(problem, history, i, init_guess[i, :], options)
    end
    return new_strat
end

function solve_mixed(
    problem::Problem;
    options = DEFAULT_OPTIONS
)
    # draw init points from log-normal distribution
    history = exp.(options.init_mu .+ options.init_sigma .* randn(problem.n, 2, options.n_init_points))
    for i in 1:options.max_iters
        last_history = copy(history)
        for t in 1:options.n_init_points
            # on each iter, solve to maximize average payoff given current history
            # replace one of the history entries with new optimum
            # why in the world doesn't julia just use 0-based indexing???
            history[:, :, t] = single_mixed_iter(problem, history, history[:, :, t], options)
        end
        # todo: pretty sure this never triggers when strategy is mixed, since it compare the entire history, which fluctuates
        if maximum((history .- last_history) ./ (last_history .+ EPSILON)) < options.tol
            if options.verbose
                println("Exited on iteration $i")
            end
            break
        end
    end
    result = sum(
        make_3d(
            SolverResult(problem, true, history[:, 1, i], history[:, 2, i], prune = false),
            problem.n
        )
        for i in 1:options.n_init_points
    )
    return result
end

    
function test()
    println("Running test on `solve.jl`...")
    prodFunc = ProdFunc([10., 10.], [0.5, 0.5], [10., 10.], [0.5, 0.5], [0., 0.])
    csf = CSF(1., 0., 0., 0.)
    problem = Problem([1., 1.], [0.01, 0.01], prodFunc, csf)
    @time solve_iters_sol = solve_iters(problem)
    @time solve_grid_sol = solve_grid(problem)
    @time solve_roots_sol = solve_roots(problem)
    @time solve_hybrid_sol = solve_hybrid(problem)
    println("With `solve_iters`:")
    print(solve_iters_sol)
    println("With `solve_grid`:")
    print(solve_grid_sol)
    println("With `solve_roots`:")
    print(solve_roots_sol)
    println("With `solve_hybrid`:")
    print(solve_roots_sol)
    return
end
