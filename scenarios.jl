using Plots, Plots.PlotMeasures
include("./solve.jl")


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


struct ScenarioResult
    scenario::Scenario
    solverResults::Array{SolverResult}
end


function mean(x; dims)
    sum(x, dims = dims) ./ size(x, dims)
end


# Functions for plotting with only single varying param

function get_values_for_plot(results::Vector{SolverResult})
    n_steps = length(results)
    n_players = size(results[1].Xs, 1)
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

function create_plot(results::Vector{SolverResult}, xaxis, xlabel, plotsize, labels, title, logscale)
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
    final_plot = plot(
        perf_plt, safety_plt, total_safety_plt, payoff_plt,
        layout = (2, 2), size = plotsize, legend_font_pointsize = 6,
        legend_background_color = RGBA(1., 1., 1., 0.5),
        left_margin = 20px
    )
    if !isnothing(title)
        title!(title)
    end
    return final_plot
end

# functions for plotting with single varying param, scatterplot instead of line plot (for when multiple solutions are found)

function get_values_for_scatterplot(results::Vector{SolverResult}, xaxis; take_avg = false)
    if !take_avg
        xaxis_ = vcat((fill(x, size(r.Xs, 1)) for (x, r) in zip(xaxis, results))...)
        Xs = vcat((r.Xs for r in results)...)
        Xp = vcat((r.Xp for r in results)...)
        s = vcat((r.s for r in results)...)
        p = vcat((r.p for r in results)...)
        payoffs = vcat((r.payoffs for r in results)...)
        total_safety = get_total_safety(s)
        return xaxis_, Xs, Xp, s, p, total_safety, payoffs
    else
        Xs = vcat((mean(r.Xs, dims = 1) for r in results)...)
        Xp = vcat((mean(r.Xp, dims = 1) for r in results)...)
        s = vcat((mean(r.s, dims = 1) for r in results)...)
        p = vcat((mean(r.p, dims = 1) for r in results)...)
        payoffs = vcat((mean(r.payoffs, dims = 1) for r in results)...)
        total_safety = vcat((mean(get_total_safety(r.s), dims = 1) for r in results)...)
        return xaxis, Xs, Xp, s, p, total_safety, payoffs
    end
end

function create_scatterplot(
    results::Vector{SolverResult}, xaxis, xlabel,
    plotsize, labels, title, logscale;
    take_avg = false
)
    (xaxis_, Xs, Xp, s, p, total_safety, payoffs) = get_values_for_scatterplot(results, xaxis, take_avg = take_avg)
    labels_ = reshape(labels, 1, :)
    Xp_plt = scatter(xaxis_, Xp, xlabel = xlabel, ylabel = "Xₚ", labels = labels_)
    Xs_plt = scatter(xaxis_, Xs, xlabel = xlabel, ylabel = "Xₛ", labels = labels_)
    perf_plt = scatter(xaxis_, p, xlabel = xlabel, ylabel = "performance", labels = labels_)
    safety_plt = scatter(xaxis_, s, xlabel = xlabel, ylabel = "safety", labels = labels_)
    if logscale
        yaxis!(Xp_plt, :log10)
        yaxis!(Xs_plt, :log10)
        yaxis!(perf_plt, :log10)
        yaxis!(safety_plt, :log10)
    end
    total_safety_plt = scatter(xaxis_, total_safety, xlabel = xlabel, ylabel = "σ", label = nothing)
    payoff_plt = scatter(xaxis_, payoffs, xlabel = xlabel, ylabel = "payoff", labels = labels_)
    final_plot = plot(
        Xp_plt, Xs_plt,
        perf_plt, safety_plt,
        total_safety_plt, payoff_plt,
        layout = (3, 2), size = (plotsize[1], plotsize[2] * 1.5), legend_font_pointsize = 6,
        legend_background_color = RGBA(1., 1., 1., 0.5),
        markeralpha = 0.25, markerstrokealpha = 0.,
        left_margin = 20px
    )
    if !isnothing(title)
        title!(title)
    end
    return final_plot
end


# Functions for plotting with secondary variation

function get_colors(n_lines)
    colors = RGB[]
    red = [1., 0., 0.]
    blue = [0., 0., 1.]
    for i in 0:(n_lines-1)
        λ = i / (n_lines - 1)
        push!(colors, RGB((red * λ + blue * (1 - λ))...))
    end
    return colors
end

function get_color_palettes(n_lines, n_players)
    palettes = ColorPalette[]
    red = [1., 0., 0.]
    green = [0., 1., 0.]
    blue = [0., 0., 1.]
    for i in 0:(n_lines-1)
        λ = i / (n_lines-1)
        color1 = RGB((red * λ + blue * (1 - λ))...)
        color2 = RGB(((red + blue) * 0.5 * λ + green * (1 - λ))...)
        push!(palettes, palette([color2, color1], n_players))
    end
    return palettes
end

function get_values_for_plot(results::Array{SolverResult, 2})
    (n_steps_secondary, n_steps) = size(results)
    n_players = size(results[1, 1].Xs, 1)
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

function are_players_same(results::Array{SolverResult, 2})
    (n_steps_secondary, n_steps) = size(results)
    n_players = size(results[1, 1].Xs)[1]
    strats = Array{Float64}(undef, n_steps_secondary, n_steps, n_players, 2)
    for i in 1:n_steps_secondary
        for j in 1:n_steps
            strats[i, j, :, 1] = results[i, j].Xs
            strats[i, j, :, 2] = results[i, j].Xp
        end
    end
    return all(
        isapprox(
            strats[:, :, 1, :] - strats[:, :, i, :],
            zeros(n_steps_secondary, n_steps, 2),
            atol=1e-2
        ) for i in 2:n_players
    )
end


function _plot_helper_same(xaxis, s, p, total_safety, payoffs, xlabel, labels)
    n_steps_secondary = size(s)[1]
    colors = get_colors(n_steps_secondary)
    combine_values(x) = mean(x, dims=2)
    perf_plt = plot(
        xaxis, combine_values(p[1, :, :]),
        xlabel = xlabel, ylabel = "performance",
        labels = labels[1],
        color = colors[1]
    )
    safety_plt = plot(
        xaxis, combine_values(s[1, :, :]),
        xlabel = xlabel, ylabel = "safety",
        labels = labels[1],
        color = colors[1]
    )
    total_safety_plt = plot(
        xaxis, total_safety[1, :],
        xlabel = xlabel, ylabel = "σ",
        label = labels[1],
        color = colors[1]
    )
    payoff_plt = plot(
        xaxis, combine_values(payoffs[1, :, :]),
        xlabel = xlabel,
        ylabel = "payoff",
        labels = labels[1],
        color = colors[1]
    )
    for i in 2:n_steps_secondary
        plot!(
            perf_plt,
            xaxis, combine_values(p[i, :, :]),
            labels = labels[i],
            color = colors[i]
        )
        plot!(
            safety_plt,
            xaxis, combine_values(s[i, :, :]),
            labels = labels[i],
            color = colors[i]
        )
        plot!(
            total_safety_plt,
            xaxis, total_safety[i, :],
            label = labels[i],
            color = colors[i]
        )
        plot!(
            payoff_plt,
            xaxis, combine_values(payoffs[i, :, :]),
            labels = labels[i],
            color = colors[i]
        )
    end
    return perf_plt, safety_plt, total_safety_plt, payoff_plt
end

function _plot_helper_het(xaxis, s, p, total_safety, payoffs, xlabel, labels)
    (n_steps_secondary, _, n_players) = size(s)
    palettes = get_color_palettes(n_steps_secondary, n_players)
    colors = get_colors(n_steps_secondary)
    labels1 = reshape(["$(labels[1]), player $i" for i in 1:n_players], 1, :)
    perf_plt = plot(
        xaxis, p[1, :, :],
        xlabel = xlabel, ylabel = "performance",
        labels = labels1,
        palette = palettes[1]
    )
    safety_plt = plot(
        xaxis, s[1, :, :],
        xlabel = xlabel, ylabel = "safety",
        labels = labels1,
        palette = palettes[1]
    )
    total_safety_plt = plot(
        xaxis, total_safety[1, :],
        xlabel = xlabel, ylabel = "σ",
        label = labels[1],
        color = colors[1]
    )
    payoff_plt = plot(
        xaxis, payoffs[1, :, :],
        xlabel = xlabel,
        ylabel = "payoff",
        labels = labels1,
        palette = palettes[1]
    )
    for i in 2:n_steps_secondary
        labelsi = reshape(["$(labels[i]), player $j" for j in 1:n_players], 1, :)
        plot!(
            perf_plt,
            xaxis, p[i, :, :],
            labels = labelsi,
            palette = palettes[i]
        )
        plot!(
            safety_plt,
            xaxis, s[i, :, :],
            labels = labelsi,
            palette = palettes[i]
        )
        plot!(
            total_safety_plt,
            xaxis, total_safety[i, :],
            label = labels[i],
            color = colors[i]
        )
        plot!(
            payoff_plt,
            xaxis, payoffs[i, :, :],
            labels = labelsi,
            palette = palettes[i]
        )
    end
    return perf_plt, safety_plt, total_safety_plt, payoff_plt
end


function create_plot(results::Array{SolverResult, 2}, xaxis, xlabel, plotsize, labels, title, logscale)
    (s, p, total_safety, payoffs) = get_values_for_plot(results)
    players_same = are_players_same(results)
    (perf_plt, safety_plt, total_safety_plt, payoff_plt) = if players_same
        _plot_helper_same(xaxis, s, p, total_safety, payoffs, xlabel, labels)
    else
        _plot_helper_het(xaxis, s, p, total_safety, payoffs, xlabel, labels)
    end
    
    if logscale
        yaxis!(perf_plt, :log10)
        yaxis!(safety_plt, :log10)
    end
    final_plot = plot(
        perf_plt, safety_plt, total_safety_plt, payoff_plt,
        layout = (2, 2), size = plotsize, legend_font_pointsize = 6,
        legend_background_color = RGBA(1., 1., 1., 0.5),
        left_margin = 20px
    )
    if !isnothing(title)
        title!(title)
    end
    return final_plot
end


function linspace(start, stop, steps, reps = 1)
    return transpose(repeat(range(start, stop=stop, length=steps), outer = (1, reps)))
end

function get_result(problem, method, options)
    if method == :hybrid
        return solve_hybrid(problem, options = options)
    elseif method == :roots
        return solve_roots(problem, ftol = options.solver_tol)
    elseif method == :scatter
        return solve_scatter(problem; options)
    elseif method == :mixed
        return solve_mixed(problem; options)
    else
        # implicitly method == :iters by default
        return solve_iters(problem; options)
    end
end


function solve_with_secondary_variation(
    scenario::Scenario,
    method,
    options
)
    if method == :scatter || method == :mixed
        println("Secondary variation is unsupported with the mixed or scatter solvers.")
        return ScenarioResult(SolverResult[], plot())
    end
    varying_param_ = getfield(scenario, scenario.varying_param)
    varying_param = size(varying_param_, 1) == 1 ? repeat(varying_param_, scenario.n_players) : varying_param_
    n_steps = size(varying_param, 2)
    secondary_varying_param = getfield(scenario, scenario.secondary_varying_param)
    n_steps_secondary = size(secondary_varying_param, 2)
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
            results[j, i] = get_result(problem, method, options)
        end
    end
    if options.verbose
        print.(results)
    end

    return ScenarioResult(scenario, results)
end


function solve(
    scenario::Scenario;
    method = :iters,
    options = DEFAULT_OPTIONS
)
    if !isnothing(scenario.secondary_varying_param)
        return solve_with_secondary_variation(
            scenario,
            method,
            options
        )
    end

    varying_param_ = getfield(scenario, scenario.varying_param)
    varying_param = size(varying_param_, 1) == 1 ? repeat(varying_param_, scenario.n_players) : varying_param_
    n_steps = size(varying_param, 2)
    
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
        results[i] = get_result(problem, method, options)
    end
    if options.verbose
        print.(results)
    end

    return ScenarioResult(scenario, results)
end


function plot_result(
    res::ScenarioResult;
    plotsize = (900, 600),
    title = nothing,
    logscale = false,
    take_avg = false
)
    varying_param = getfield(res.scenario, res.scenario.varying_param)
    n_steps = size(varying_param, 2)
    xaxis = if size(varying_param, 1) == 1
        reshape(varying_param, :, 1)
    elseif varying_param[1, :] == varying_param[2, :]
        varying_param[1, :]
    else
        1:n_steps
    end
    plt = if isnothing(res.scenario.secondary_varying_param)
        labels = ["player $i" for i in 1:res.scenario.n_players]
        if ndims(res.solverResults[1].Xs) > 1
            create_scatterplot(
                res.solverResults,
                xaxis,
                res.scenario.varying_param,
                plotsize, 
                labels,
                title,
                logscale,
                take_avg = take_avg
            )
        else
            create_plot(
                res.solverResults,
                xaxis,
                res.scenario.varying_param,
                plotsize,
                labels,
                title,
                logscale
            )
        end
    else
        secondary_varying_param = getfield(res.scenario, res.scenario.secondary_varying_param)
        n_steps_secondary = size(secondary_varying_param, 2)
        labels = ["$(res.scenario.secondary_varying_param) = $(secondary_varying_param[:, i])" for i in 1:n_steps_secondary]
        create_plot(
            res.solverResults,
            xaxis,
            res.scenario.varying_param,
            plotsize,
            labels,
            title,
            logscale
        )
    end

    return plt
end


function test(method = :hybrid)
    println("Running test on `scenarios.jl`...")

    A = [10., 10.]
    α = [0.5, 0.75]
    B = [10., 10.]
    β = [0.5, 0.5]
    θ = transpose([0. 0.; 0.5 0.5])
    d = [1., 1.]
    r = linspace(0.01, 0.1, Threads.nthreads(), 2)

    scenario = Scenario(2, A, α, B, β, θ, d, r, secondary_varying_param = :θ)

    @time res = solve(scenario, method = method, options = IterOptions(verbose = true))
    plot_result(res)
end
