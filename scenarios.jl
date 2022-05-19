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
    csf::CSF
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
) = Scenario(n_players, A, α, B, β, θ, d, r, CSF(w, l, a_w, a_l), varying_param, secondary_varying_param)

function get_values_for_plot(results::Array{SolverResult, 1})
    n_steps = length(results)
    n_players = size(results[1].strats)[1]
    s = Array{Float64}(undef, n_steps, n_players)
    p = similar(s)
    payoffs = similar(s)
    for (i, r) in enumerate(results)
        s[i, :] = r.s
        p[i, :] = r.p
        payoffs[i, :] = r.payoffs
    end
    total_safety = get_total_safety(s)
    return s, p, total_safety, payoffs
end

function create_plot(results::Array{SolverResult, 1}, xaxis, xlabel, plotname, labels, title, logscale)
    (s, p, total_safety, payoffs) = get_values_for_plot(results)
    labels_ = reshape(labels, 1, :)
    perf_plt = plot(xaxis, p, xlabel = xlabel, ylabel = "performance", labels = labels_)
    safety_plt = plot(xaxis, s, xlabel = xlabel, ylabel = "safety", labels = labels_)
    if logscale
        yaxis!(perf_plt, :log10)
        yaxis!(safety_plt, :log10)
    end
    total_safety_plt = plot(xaxis, total_safety, xlabel = xlabel, ylabel = "σ", label = nothing)
    payoff_plt = plot(xaxis, payoffs, xlabel = xlabel, ylabel = "payoff", labels = labels_)
    plot(
        perf_plt, safety_plt, total_safety_plt, payoff_plt,
        layout = (2, 2), size = (1200, 800), legend_font_pointsize = 6
    )
    if title != nothing
        title!(title)
    end
    if plotname != nothing
        Plots.savefig("$(plotname).png")
    else
        gui()
    end
end

function get_values_for_plot(results::Array{SolverResult, 2})
    (n_steps_secondary, n_steps) = size(results)
    n_players = size(results[1, 1].strats)[1]
    s = Array{Float64}(undef, n_steps_secondary, n_steps, n_players)
    p = similar(s)
    payoffs = similar(s)
    for i in 1:n_steps_secondary
        for j in 1:n_steps
            s[i, j, :] = results[i, j].s
            p[i, j, :] = results[i, j].p
            payoffs[i, j, :] = results[i, j].payoffs
        end
    end
    total_safety = get_total_safety(s)
    return s, p, total_safety, payoffs
end

function create_plot(results::Array{SolverResult, 2}, xaxis, xlabel, plotname, labels, title, logscale)
    (s, p, total_safety, payoffs) = get_values_for_plot(results)
    (n_steps_secondary, _, n_players) = size(s)
    labels1 = reshape(["$(labels[1]), player $i" for i in 1:n_players], 1, :)
    perf_plt = plot(xaxis, p[1, :, :], xlabel = xlabel, ylabel = "performance", labels = labels1)
    safety_plt = plot(xaxis, s[1, :, :], xlabel = xlabel, ylabel = "safety", labels = labels1)
    total_safety_plt = plot(xaxis, total_safety[1, :], xlabel = xlabel, ylabel = "σ", label = labels[1])
    payoff_plt = plot(xaxis, payoffs[1, :, :], xlabel = xlabel, ylabel = "payoff", labels = labels1)
    for i in 2:n_steps_secondary
        labelsi = reshape(["$(labels[i]), player $j" for j in 1:n_players], 1, :)
        plot!(perf_plt, xaxis, p[i, :, :], labels = labelsi)
        plot!(safety_plt, xaxis, s[i, :, :], labels = labelsi)
        plot!(total_safety_plt, xaxis, total_safety[i, :], label = labels[i])
        plot!(payoff_plt, xaxis, payoffs[i, :, :], labels = labelsi)
    end
    if logscale
        yaxis!(perf_plt, :log10)
        yaxis!(safety_plt, :log10)
    end
    plot(
        perf_plt, safety_plt, total_safety_plt, payoff_plt,
        layout = (2, 2), size = (1200, 800), legend_font_pointsize = 6
    )
    if title != nothing
        title!(title)
    end
    if plotname != nothing
        Plots.savefig("$(plotname).png")
    else
        gui()
    end
end


function linspace(start, stop, steps, reps = 1)
    stepsize = (stop - start) / (steps - 1)
    return transpose(repeat(start:stepsize:stop, outer = (1, reps)))
end

function get_result(problem, max_iters, tol, verbose, solve_method)
    if solve_method == :iters
        return solve_iters(
            problem,
            max_iters = max_iters,
            tol = tol,
            verbose = verbose
        )
    elseif solve_method == :roots
        return solve_roots(problem)
    else
        # implicitly solve_method == :hybrid by default
        return solve_hybrid(
            problem,
            max_iters = max_iters,
            tol = tol,
            verbose = verbose
        )
    end
end


function solve_with_secondary_variation(
    scenario::Scenario,
    makeplot,
    plotname,
    title,
    logscale,
    solve_method,
    max_iters,
    tol,
    verbose
)
    varying_param_ = getfield(scenario, scenario.varying_param)
    varying_param = size(varying_param_)[1] == 1 ? repeat(varying_param_, scenario.n_players) : varying_param_
    n_steps = size(varying_param)[2]
    secondary_varying_param = getfield(scenario, scenario.secondary_varying_param)
    n_steps_secondary = size(secondary_varying_param)[2]
    # create stacks of variables to send to solver
    # everything needs to have shape n_players x n_steps x n_steps_secondary
    A = similar(scenario.A, (scenario.n_players, n_steps, n_steps_secondary))
    α = similar(scenario.α, (scenario.n_players, n_steps, n_steps_secondary))
    B = similar(scenario.B, (scenario.n_players, n_steps, n_steps_secondary))
    β = similar(scenario.β, (scenario.n_players, n_steps, n_steps_secondary))
    θ = similar(scenario.θ, (scenario.n_players, n_steps, n_steps_secondary))
    d = similar(scenario.d, (scenario.n_players, n_steps, n_steps_secondary))
    r = similar(scenario.r, (scenario.n_players, n_steps, n_steps_secondary))
    for (newvar, symbol) in zip((A, α, B, β, θ, d, r), (:A, :α, :B, :β, :θ, :d, :r))
        if symbol == scenario.varying_param
            copyto!(
                newvar,
                repeat(varying_param, outer = (1, n_steps_secondary))
            )
        elseif symbol == scenario.secondary_varying_param
            copyto!(
                newvar,
                repeat(secondary_varying_param, inner = (1, n_steps))
            )
        else
            var = getfield(scenario, symbol)
            copyto!(
                newvar,
                repeat(var, inner = (1, n_steps), outer = (1, n_steps_secondary))
            )
        end
    end
    results = Array{SolverResult}(undef, (n_steps_secondary, n_steps))
    Threads.@threads for i in 1:n_steps
        Threads.@threads for j in 1:n_steps_secondary
            prodFunc = ProdFunc(A[:, i, j], α[:, i, j], B[:, i, j], β[:, i, j], θ[:, i, j])
            problem = Problem(d[:, i, j], r[:, i, j],  prodFunc, scenario.csf)
            results[j, i] = get_result(problem, max_iters, tol, verbose, solve_method)
        end
    end

    if makeplot
        xaxis = varying_param[1, :] == varying_param[2, :] ? varying_param[1, :] : 1:n_steps
        labels = ["$(scenario.secondary_varying_param) = $(secondary_varying_param[:, i])" for i in 1:n_steps_secondary]
        create_plot(
            results,
            xaxis,
            scenario.varying_param,
            plotname,
            labels,
            title,
            logscale
        )
    end

    return results
end


function solve(
    scenario::Scenario;
    makeplot = true,
    plotname = nothing,  # if set to string, will save fig instead of displaying interactive
    title = nothing,
    logscale = false,
    solve_method = :hybrid,
    max_iters = DEFAULT_ITER_MAX_ITERS,
    tol = DEFAULT_ITER_TOL,
    verbose = false
)
    if scenario.secondary_varying_param != nothing
        return solve_with_secondary_variation(
            scenario,
            makeplot,
            plotname,
            title,
            logscale,
            solve_method,
            max_iters,
            tol,
            verbose
        )
    end

    varying_param_ = getfield(scenario, scenario.varying_param)
    varying_param = size(varying_param_)[1] == 1 ? repeat(varying_param_, scenario.n_players) : varying_param_
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
        if symbol == scenario.varying_param
            copy!(newvar, varying_param)
        else
            var = getfield(scenario, symbol)
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

    results = Array{SolverResult}(undef, n_steps)
    # send to solver
    Threads.@threads for i in 1:n_steps
        prodFunc = ProdFunc(A[:, i], α[:, i], B[:, i], β[:, i], θ[:, i])
        problem = Problem(d[:, i], r[:, i],  prodFunc, scenario.csf)
        results[i] = get_result(problem, max_iters, tol, verbose, solve_method)
    end

    if makeplot
        xaxis = varying_param[1, :] == varying_param[2, :] ? varying_param[1, :] : 1:n_steps
        labels = ["player $i" for i in 1:n_steps]
        create_plot(results, xaxis, scenario.varying_param, plotname, labels, title, logscale)
    end

    return results
end


function test(solve_method = :hybrid)
    println("Running test on `scenarios.jl`...")

    A = [10., 10.]
    α = [0.5, 0.75]
    B = [10., 10.]
    β = [0.5, 0.5]
    θ = transpose([0. 0.; 0.5 0.5])
    d = [1., 1.]
    r = linspace(0.01, 0.1, Threads.nthreads(), 2)

    scenario = Scenario(2, A, α, B, β, θ, d, r, secondary_varying_param = :θ)

    @time result = solve(scenario, solve_method = solve_method, verbose = true)
end
