const EPSILON = 1e-8  # small number to help numerical stability in some places

Base.@kwdef struct SolverOptions{T <: AbstractFloat}
    tol::T = 1e-6
    max_iters::Int = 100
    iter_algo::Optim.AbstractOptimizer = NelderMead()
    iter_options::Optim.Options = Optim.Options(x_tol = 1e-8, iterations = 500)
    init_guess::Union{T, Nothing} = nothing  # if nothing, then intialize randomly
    n_points::Int = 4
    init_mu::T = -1.
    init_sigma::T = 0.1
    verbose::Bool = false
    verify::Bool = true
    verify_mult::T = 1.1
    retries::Int = 10
end

function SolverOptions(options::SolverOptions; kwargs...)
    fields = ((f in keys(kwargs)) ? kwargs[f] : getfield(options, f) for f in fieldnames(SolverOptions))
    return SolverOptions(fields...)
end

function get_init_guess(options::SolverOptions, shape)
    if isnothing(options.init_guess)
        return exp.(options.init_mu .+ options.init_sigma .* randn(shape))
    else
        return fill(options.init_guess, shape)
    end
end

function check_symmetric(problem, strat, options)
    if (
        is_symmetric(problem)
        && !all(isapprox.(strat[1, :]', strat[2:end, :], rtol = sqrt(options.tol)))
    )
        if options.verbose
            println("Solution should be symmetric but is not")
        end
        return false
    end

    return true
end

# Check some points around the given strat
# returns true if strat satisfies optimality (in nash eq. sense) at those points
# won't always catch wrong solutions, but does a decent job
function verify_i(problem, i, Xs, Xp, payoffs, options)
    higher_Xs = setindex!(
        copy(Xs),
        (Xs[i] == 0) ? EPSILON : options.verify_mult * Xs[i],
        i
    )
    higher_Xp = setindex!(
        copy(Xp),
        (Xp[i] == 0) ? EPSILON : options.verify_mult * Xp[i],
        i
    )
    lower_Xs = setindex!(copy(Xs), Xs[i] * options.verify_mult, i)
    lower_Xp = setindex!(copy(Xp), Xp[i] * options.verify_mult, i)

    payoff_higher_Xs = get_payoff(problem, i, higher_Xs, Xp)
    payoff_higher_Xp = get_payoff(problem, i, Xs, higher_Xp)
    payoff_lower_Xs = get_payoff(problem, i, lower_Xs, Xp)
    payoff_lower_Xp = get_payoff(problem, i, Xs, lower_Xp)
    
    mirror_Xs = setindex!(
        copy(Xs),
        (sum(Xs) - Xs[i]) / (length(Xs) - 1),
        i
    )
    mirror_Xp = setindex!(
        copy(Xp),
        (sum(Xp) - Xp[i]) / (length(Xp) - 1),
        i
    )
    payoff_mirror = get_payoff(problem, i, mirror_Xs, mirror_Xp)
    if any(is_napprox_greater.(
        (
            payoff_higher_Xs,
            payoff_higher_Xp,
            payoff_lower_Xs,
            payoff_lower_Xp,
            payoff_mirror
        ),
        payoffs[i],
        rtol = sqrt(options.tol)
    ))
        if options.verbose
            println("Solution failed verification!")
            println("| Xs = $Xs, Xp = $Xp: $(payoffs[i])")
            println("| Xs[$i] = $(higher_Xs[i]): $payoff_higher_Xs")
            println("| Xp[$i] = $(higher_Xp[i]): $payoff_higher_Xp")
            println("| Xs[$i] = $(lower_Xs[i]): $payoff_lower_Xs")
            println("| Xp[$i] = $(lower_Xp[i]): $payoff_lower_Xp")
            println("∟ Symmetric: $payoff_mirror")
        end
        return false
    end

    return true
end

function verify(problem, strat, options)
    if !check_symmetric(problem, strat, options)
        return false
    end

    Xs = strat[:, 1]
    Xp = strat[:, 2]
    payoffs = get_payoffs(problem, Xs, Xp)
    for i in 1:problem.n
        if !verify_i(problem, i, Xs, Xp, payoffs, options)
            return false
        end
    end
    return true
end

function verify(problem::ProblemWithBeliefs, strat, options)
    if !check_symmetric(problem, strat, options)
        return false
    end

    Xs = strat[:, 1]
    Xp = strat[:, 2]
    payoffs = [get_payoff(problem.beliefs[i], i, Xs, Xp) for i in 1:problem.n]
    for i in 1:problem.n
        if !verify_i(problem.beliefs[i], i, Xs, Xp, payoffs, options)
            return false
        end
    end
    return true
end

# ITERATING METHOD `solve_iters`

function single_iter_for_i(
    problem, strat, i, init_guess,
    options = SolverOptions()
)
    obj = get_func(problem, i, strat)
    obj_(x) = obj(exp.(x))
    try
        res = optimize(
            obj_,
            log.(init_guess),
            options.iter_algo,
            options.iter_options
        )
        return exp.(Optim.minimizer(res))
    catch e
        if isa(e, InterruptException)
            throw(e)
        end
        if options.verbose
            println("Warning: Encountered $e, returning NaN")
        end
        return [NaN, NaN]
    end
end

function solve_single_iter(
    problem::Problem, strat::Array,
    options = SolverOptions()
)
    new_strats = similar(strat)
    for i in 1:problem.n
        new_strats[i, :] = single_iter_for_i(problem, strat, i, strat[i, :], options)
    end
    return new_strats
end

function suggest_iter_strat(problem, init_guess, options)
    strat = init_guess
    for i in 1:options.max_iters
        new_strat = solve_single_iter(problem, strat, options)
        if all(isapprox.(
            log.(new_strat), log.(strat),
            atol = EPSILON, rtol = options.tol
        ))
            if options.verbose
                println("Converged on iteration $i")
            end
            return true, new_strat
        end
        strat = new_strat
    end
    if options.verbose
        println("Reached max iterations")
    end
    return false, strat
end

function solve_iters(
    problem,
    options = SolverOptions()
)
    init_guess = get_init_guess(options, (problem.n, 2))
    converged, strat = suggest_iter_strat(problem, init_guess, options)
    success = converged && (!options.verify || verify(problem, strat, options))
    if success || options.retries == 0
        return SolverResult(problem, success, strat[:, 1], strat[:, 2])
    else
        if options.verbose
            println("Retrying...")
        end
        return solve_iters(
            problem,
            SolverOptions(options, retries = options.retries - 1)
        )
    end
end

# SCATTER ITERATING METHOD:
# Runs iter_solve from multiple init points and compares results

function solve_scatter(
    problem::Problem,
    options = SolverOptions()
)
    # draw init points from log-normal distribution
    results = SolverResult[]
    Threads.@threads for i in 1:options.n_points
        result = solve_iters(
            problem,
            options
        )
        if result.success
            push!(results, result)
        end
    end
    if length(results) == 0
        println("None of the solver iterations converged.")
        return [get_null_result(problem.n)]
    end
    return results
end


# Solver for mixed strategy equilibria
# Runs iterating solver over history of run

function single_mixed_iter_for_i(problem, history, i, init_guess, options = SolverOptions())
    objs = [get_func(problem, i, history[:, :, j]) for j in axes(history, 3)]
    obj_(x) = sum(obj(exp.(x)) for obj in objs)
    # if init_guess is zero-effort, try breaking out
    if all(init_guess .== 0.)
        init_guess = exp.(options.init_mu .+ options.init_sigma .* randn(2))
    end
    try
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
    catch e
        if isa(e, InterruptException)
            throw(e)
        end
        if options.verbose
            println("Warning: Encountered $e, returning NaN")
        end
        return [NaN, NaN]
    end
end

function single_mixed_iter(problem, history, init_guess, options = SolverOptions())
    new_strat = Array{Float64}(undef, problem.n, 2)
    for i in 1:problem.n
        new_strat[i, :] = single_mixed_iter_for_i(problem, history, i, init_guess[i, :], options)
    end
    return new_strat
end

function solve_mixed(
    problem::Problem,
    options = SolverOptions()
)
    # draw init points from log-normal distribution
    history = exp.(options.init_mu .+ options.init_sigma .* randn(problem.n, 2, options.n_points))
    for i in 1:options.max_iters
        last_history = copy(history)
        for t in 1:options.n_points
            # on each iter, solve to maximize average payoff given current history
            # replace one of the history entries with new optimum
            history[:, :, t] = single_mixed_iter(problem, history, history[:, :, t], options)
        end
        # todo: this might never trigger when strategy is mixed, since it compares the entire history, which fluctuates
        if all(isapprox.(
            log.(history), log.(last_history),
            atol = EPSILON, rtol = options.tol
        ))
            if options.verbose
                println("Exited on iteration $i")
            end
            break
        end
    end
    results = [
        SolverResult(problem, true, history[:, 1, i], history[:, 2, i])
        for i in 1:options.n_points
    ]
    return results
end


# HYBRID METHOD (Runs solve_iter, then solve_mixed if that is unsuccessful. Finds only pure strategies.)

function attempt_mixed(problem, options)
    mixed_sol = solve_mixed(problem, SolverOptions(options))
    if (
        mixed_sol.success
        && slices_approx_equal(log.(mixed_sol.Xs), 1, EPSILON, sqrt(options.tol))
        && slices_approx_equal(log.(mixed_sol.Xp), 1, EPSILON, sqrt(options.tol))
    )
        Xs = mean(hcat((sol.Xs for sol in mixed_sol)...), 1)
        Xp = mean(hcat(sol.Xp for sol in mixed_sol)..., 1)
        return SolverResult(problem, true, Xs, Xp)
    elseif options.retries > 0
        if options.verbose
            println("solve_mixed failed. Retrying...")
        end
        return attempt_mixed(problem, SolverOptions(options, retries = options.retries - 1))
    else
        if options.verbose
            println("solve_mixed also failed")
            println("| solution was Xs = $(mixed_sol.Xs), Xp = $(mixed_sol.Xp)")
            println("∟ returning null result")
        end
        return get_null_result(problem.n)
    end
end

function solve_hybrid(
    problem::Problem,
    options = SolverOptions(n_points = 2)
)
    iter_sol = solve_iters(problem, options)
    if iter_sol.success
        return iter_sol
    end

    if options.verbose
        println("solve_iters failed, trying mixed solver")
    end
    return attempt_mixed(problem, options)
end


# general purpose solver function
function solve(problem::Problem, method::Symbol, options)
    solve_func = if method == :iters
        solve_iters
    elseif method == :roots
        solve_roots
    elseif method == :scatter
        solve_scatter
    elseif method == :mixed
        solve_mixed
    elseif method == :hybrid
        solve_hybrid
    else
        throw(ArgumentError("Invalid method $method"))
    end
    return solve_func(problem, options)
end

function solve(
    problem::AbstractProblem;
    method::Symbol = :iters,
    kwargs...
)
    options = SolverOptions(SolverOptions(); kwargs...)
    return solve(problem, method::Symbol, options)
end

## Variation of solver for ProblemWithBeliefs
# Just need to have single_iter_for_i use each player's beliefs

function solve_single_iter(
    problem::ProblemWithBeliefs, strat::Array,
    options = SolverOptions()
)
    new_strats = similar(strat)
    for i in 1:problem.n
        new_strats[i, :] = single_iter_for_i(problem.beliefs[i], strat, i, strat[i, :], options)
    end
    return new_strats
end

function solve(problem::ProblemWithBeliefs, method::Symbol, options)
    @assert method == :iters "Currently only :iters is supported when solving ProblemWithBeliefs"
    solve_iters(problem, options)
end
