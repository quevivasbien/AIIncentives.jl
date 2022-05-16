using Plots
include("./model.jl")


struct Scenario
    n_players::Integer
    A::Array
    α::Array
    B::Array
    β::Array
    θ::Array
    d::Array
    r::Array
    w::Number
    l::Number
    a_w::Number
    a_l::Number
    varying_param
    secondary_varying_param
end

Scenario(
    n_players,
    A, α, B, β, θ,
    d, r;
    w = 1., l = 0., a_w = 0., a_l = 0.,
    varying_param = :r,
    secondary_varying_param = nothing
) = Scenario(n_players, A, α, B, β, θ, d, r, w, l, a_w, a_l, varying_param, secondary_varying_param)

function get_values_for_plot(results::Array)
    n_steps = length(results)
    n_players = size(results[1].strats)[1]
    s = similar(results[1].s, n_steps, n_players)
    p = similar(results[1].p, n_steps, n_players)
    payoffs = similar(results[1].payoffs, n_steps, n_players)
    for (i, r) in enumerate(results)
        s[i, :] = r.s
        p[i, :] = r.p
        payoffs[i, :] = r.payoffs
    end
    total_safety = prod(s ./ (1. .+ s), dims = 2)
    return s, p, total_safety, payoffs
end

function create_plot(results::Array, xaxis, xlabel, plotname, labels, title, logscale)
    (s, p, total_safety, payoffs) = get_values_for_plot(results)
    perf_plt = plot(xaxis, p, xlabel = xlabel, ylabel = "performance")
    safety_plt = plot(xaxis, s, xlabel = xlabel, ylabel = "safety")
    if logscale
        yaxis!(perf_plt, :log10)
        yaxis!(safety_plt, :log10)
    end
    total_safety_plt = plot(xaxis, total_safety, xlabel = xlabel, ylabel = "σ")
    payoff_plt = plot(xaxis, payoffs, xlabel = xlabel, ylabel = "payoff")
    plot(perf_plt, safety_plt, total_safety_plt, payoff_plt, layout = (2, 2))
    if title != nothing
        title!(title)
    end
    gui()
end


function linspace(start, stop, steps, reps = 1)
    stepsize = (stop - start) / (steps - 1)
    return transpose(repeat(start:stepsize:stop, outer = (1, reps)))
end


function solve_with_secondary_variation(
    scenario::Scenario,
    plot,
    plotname,
    labels,
    title,
    logscale,
    solve_method,
    max_iters,
    tol
)
    # To-Do
    println("Not yet implemented!")
    return(get_null_result())
end


function solve(
    scenario::Scenario;
    makeplot = true,
    plotname = "scenario",
    labels = nothing,
    title = nothing,
    logscale = false,
    solve_method = :hybrid,
    max_iters = DEFAULT_ITER_MAX_ITERS,
    tol = DEFAULT_ITER_TOL,
    verbose = false
)
    if scenario.secondary_varying_param != nothing
        return solve_with_secondary_variation(scenario, plot, plotname, labels, title, logscale, solve_method, max_iters, tol)
    end

    if labels != nothing && length(labels) != scenario.n_players
        println("Length of labels should match number of players")
        return get_null_result()
    end

    varying_param = getfield(scenario, scenario.varying_param)
    n_steps = size(varying_param)[2]
    
    A = similar(scenario.A, (scenario.n_players, n_steps))
    α = similar(scenario.α, (scenario.n_players, n_steps))
    B = similar(scenario.B, (scenario.n_players, n_steps))
    β = similar(scenario.β, (scenario.n_players, n_steps))
    θ = similar(scenario.θ, (scenario.n_players, n_steps))
    d = similar(scenario.d, (scenario.n_players, n_steps))
    r = similar(scenario.r, (scenario.n_players, n_steps))

    # create stacks of variables to send to solver
    # everything needs to have shape n_players x n_steps
    for (newvar, symbol) in zip((A, α, B, β, θ, d, r), (:A, :α, :B, :β, :θ, :d, :r))
        var = getfield(scenario, symbol)
        if symbol == scenario.varying_param
            copy!(newvar, var)
        else
            copy!(
                newvar,
                reshape(
                    repeat(
                        reshape(var, :), n_steps
                    ),
                    size(var)..., n_steps
                )
            )
        end
    end

    results = Array{Any}(nothing, n_steps)
    csf = CSF(scenario.w, scenario.l, scenario.a_w, scenario.a_l)
    # send to solver
    Threads.@threads for i in 1:n_steps
        prodFunc = ProdFunc(A[:, i], α[:, i], B[:, i], β[:, i], θ[:, i])
        problem = Problem(d[:, i], r[:, i],  prodFunc, csf)
        if solve_method == :iters
            results[i] = solve_iters(
                problem,
                max_iters = max_iters,
                tol = tol,
                verbose = verbose
            )
        elseif solve_method == :roots
            results[i] = solve_roots(problem)
        else
            # implicitly solve_method == :hybrid by default
            results[i] = solve_hybrid(
                problem,
                max_iters = max_iters,
                tol = tol,
                verbose = verbose
            )
        end
    end

    if makeplot
        xaxis = varying_param[1, :] == varying_param[2, :] ? varying_param[1, :] : 1:n_steps
        create_plot(results, xaxis, scenario.varying_param, plotname, labels, title, logscale)
    end

    return results
end


function test(solve_method = :hybrid)
    println("Running test on `scenarios.jl`...")

    A = [10., 10.]
    α = [0.5, 0.5]
    B = [10., 10.]
    β = [0.5, 0.5]
    θ = [0.5, 0.5]
    d = [1., 1.]
    r = linspace(0.01, 0.1, Threads.nthreads(), 2)

    scenario = Scenario(2, A, α, B, β, θ, d, r)

    @time result = solve(scenario, solve_method = solve_method, verbose = true)
end
