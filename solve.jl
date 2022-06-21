using Optim
using NLsolve

include("./Problem.jl")

const EPSILON = 1e-8  # small number to help numerical stability in some places

struct SolverOptions
    tol::Number
    max_iters::Integer
    iter_algo::Optim.AbstractOptimizer
    iter_options::Optim.Options
    init_guess::Number
    n_init_points::Integer
    init_mu::Number
    init_sigma::Number
    verbose::Bool
    verify::Integer
end

SolverOptions(
    ;
    tol::Number = 1e-6, max_iters::Integer = 100,
    iter_algo::Optim.AbstractOptimizer = LBFGS(),
    iter_options::Optim.Options = Optim.Options(x_tol = 1e-8, iterations = 500),
    init_guess::Number = 1., n_init_points::Integer = 20,
    init_mu::Number = 0., init_sigma::Number = 1.,
    verbose::Bool = false, verify::Integer = 4
) = SolverOptions(
    tol, max_iters,
    iter_algo, iter_options,
    init_guess, n_init_points,
    init_mu, init_sigma,
    verbose, verify
)

const DEFAULT_OPTIONS = SolverOptions()

# ITERATING METHOD `solve_iters`

function single_iter_for_i(
    problem, strat, i, init_guess,
    options = DEFAULT_OPTIONS
)
    obj = get_func(problem, i, strat)
    obj_(x) = obj(exp.(x))
    # TODO: the gradients I was calculating were wrong!! Fix this or just leave them off I suppose
    # jac = get_jac(problem, i, strat, inplace = false)
    # jac_!(var, x) = copy!(var, exp.(x) .* jac(exp.(x)))
    # if init_guess is zero-effort, try breaking out
    if all(init_guess .== 0.)
        init_guess = exp.(options.init_mu .+ options.init_sigma .* randn(2))
    end
    res = optimize(
        obj_,
        # jac_!,
        log.(init_guess),
        options.iter_algo,
        options.iter_options
    )
    # check the zero-input payoff
    if obj([0., 0.]) > -Optim.minimum(res)
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

function verify(problem, strat, options)
    # Checks some points around the given strat
    # returns true if strat satisfies optimality (in nash eq. sense) at those points
    # won't always catch wrong solutions, but does a decent job
    Xs = strat[:, 1]
    Xp = strat[:, 2]
    payoffs = all_payoffs(problem, Xs, Xp)
    for i in 1:problem.n, j in 1:options.verify
        higher_Xs = Xs; higher_Xs[i] *= exp(10*j*options.tol)
        higher_Xp = Xp; higher_Xp[i] *= exp(10*j*options.tol)
        lower_Xs = Xs; lower_Xs[i] *= exp(-10*j*options.tol)
        lower_Xp = Xp; lower_Xp[i] *= exp(-10*j*options.tol)
        if (
            payoff(problem, i, higher_Xs, Xp) > payoffs[i]
            || payoff(problem, i, Xs, higher_Xp) > payoffs[i]
            || payoff(problem, i, lower_Xs, Xp) > payoffs[i]
            || payoff(problem, i, Xs, lower_Xp) > payoffs[i]
        )
            if options.verbose
                println("Solution failed verification!")
            end
            return false
        end
    end
    return true
end
    
function solve_iters(
    problem::Problem,
    init_guess::Array,
    options = DEFAULT_OPTIONS
)
    strat = init_guess
    for t in 1:options.max_iters
        new_strat = solve_iters_single(problem, strat, options)
        if maximum(abs.(new_strat - strat) ./ (strat .+ EPSILON)) < options.tol
            if options.verbose
                println("Exited on iteration $t")
            end
            success = options.verify == 0 || verify(problem, new_strat, options)
            return SolverResult(
                problem,
                success,
                new_strat[:, 1], new_strat[:, 2],
                prune = false
            )
        end
        strat = new_strat
    end
    if options.verbose
        println("Reached max iterations")
    end
    return SolverResult(problem, false, strat[:, 1], strat[:, 2])
end

function solve_iters(
    problem::Problem,
    options = DEFAULT_OPTIONS
)
    return solve_iters(problem, fill(options.init_guess, (problem.n, 2)), options)
end


# SCATTER ITERATING METHOD:
# Runs iter_solve from multiple init points and compares results

function solve_scatter(
    problem::Problem,
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
    problem::Problem,
    options = DEFAULT_OPTIONS;
    f_tol = 1e-8,
    init_guesses::Vector{Float64} = [10.0^(3*i) for i in -2:2],
    resolve_multiple = true
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
            ftol = f_tol,
            method = :trust_region  # this is just the default
        )
        solution = reshape(exp.(res.zero), (problem.n, 2))
        if res.f_converged && (options.verify == 0 || verify(problem, solution, options))
            successes[i] = true
            results[i, :, :] = solution
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
    problem::Problem,
    options = DEFAULT_OPTIONS;
    init_guesses::Vector{Float64} = [10.0^(3*i) for i in -2:2]
)
    if options.verbose
        println("Finding roots...")
    end
    roots_sol = solve_roots(problem, options, init_guesses = init_guesses, resolve_multiple = false)
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
        iter_sol = solve_iters(problem, strats_, options)
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
    # jacs = [get_jac(problem, i, history[:, :, j], inplace = false) for j in size(history, 3)]
    # jac_!(var, x) = copy!(var, sum(jac(exp.(x)) for jac in jacs))
    # if init_guess is zero-effort, try breaking out
    if all(init_guess .== 0.)
        init_guess = exp.(options.init_mu .+ options.init_sigma .* randn(2))
    end
    res = optimize(
        obj_,
        # jac_!,
        log.(init_guess),
        options.iter_algo,
        options.iter_options
    )
    # check the zero-input payoff
    zero_payoff = sum(obj([0., 0.]) for obj in objs)
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
    problem::Problem,
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

    
function test_solve()
    println("Running test on `solve.jl`...")
    prodFunc = ProdFunc([10., 10.], [0.5, 0.5], [10., 10.], [0.5, 0.5], [0.25, 0.25])
    csf = CSF(1., 0., 0., 0.)
    problem = Problem([1., 1.], [0.01, 0.01], prodFunc, csf)
    options = SolverOptions(verbose = true)
    println("With `solve_iters`:")
    @time solve_iters_sol = solve_iters(problem, options)
    print(solve_iters_sol)
    println("With `solve_roots`:")
    @time solve_roots_sol = solve_roots(problem, options)
    print(solve_roots_sol)
    println("With `solve_hybrid`:")
    @time solve_hybrid_sol = solve_hybrid(problem, options)
    print(solve_roots_sol)
    println("With `solve_mixed`:")
    @time solve_mixed_sol = solve_mixed(problem, options)
    print(solve_mixed_sol)
    return
end
