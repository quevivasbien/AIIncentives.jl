# Functions for plotting with only single varying param

function get_values_for_plot(results::Vector{SolverResult}; exclude_failed = true)
    n_steps = length(results)
    n_players = size(results[1].Xs, 1)
    Xs = fill(NaN, n_steps, n_players)
    Xp = fill(NaN, n_steps, n_players)
    s = fill(NaN, n_steps, n_players)
    p = fill(NaN, n_steps, n_players)
    payoffs = fill(NaN, n_steps, n_players)
    total_safety = fill(NaN, n_steps)
    for (i, r) in enumerate(results)
        if r.success || !exclude_failed
            Xs[i, :] = r.Xs
            Xp[i, :] = r.Xp
            s[i, :] = r.s
            p[i, :] = r.p
            payoffs[i, :] = r.payoffs
            total_safety[i] = r.σ
        end
    end
    return Xs, Xp, s, p, total_safety, payoffs
end

function create_plots(results::Vector{SolverResult}, xaxis, xlabel, labels; exclude_failed = true)
    (Xs, Xp, s, p, total_safety, payoffs) = get_values_for_plot(results; exclude_failed)
    labels_ = reshape(labels, 1, :)
    Xp_plt = plot(xaxis, Xp, xlabel = xlabel, ylabel = "Xₚ", labels = labels_)
    Xs_plt = plot(xaxis, Xs, xlabel = xlabel, ylabel = "Xₛ", labels = labels_)
    perf_plt = plot(xaxis, p, xlabel = xlabel, ylabel = "performance", labels = labels_)
    safety_plt = plot(xaxis, s, xlabel = xlabel, ylabel = "safety", labels = labels_)
    total_safety_plt = plot(xaxis, total_safety, xlabel = xlabel, ylabel = "σ", label = nothing)
    payoff_plt = plot(xaxis, payoffs, xlabel = xlabel, ylabel = "payoff", labels = labels_)
    return Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt
end

function create_plot(results::Vector{SolverResult}, xaxis, xlabel, plotsize, labels, title, logscale; exclude_failed = true)
    Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt = create_plots(results, xaxis, xlabel, labels; exclude_failed)
    if logscale
        yaxis!(Xp_plt, :log10)
        yaxis!(Xs_plt, :log10)
        yaxis!(perf_plt, :log10)
        yaxis!(safety_plt, :log10)
    end
    final_plot = plot(
        Xp_plt, Xs_plt,
        perf_plt, safety_plt,
        total_safety_plt, payoff_plt,
        layout = (3, 2), size = plotsize, legend_font_pointsize = 6,
        legend_background_color = RGBA(1., 1., 1., 0.5),
        left_margin = 20px
    )
    if !isnothing(title)
        plot!(plot_title = title)
    end
    return final_plot
end

# functions for plotting with single varying param, scatterplot instead of line plot (for when multiple solutions are found)

function get_values_for_scatterplot(results::Vector{Vector{SolverResult}}, xaxis; take_avg = false, exclude_failed = true)
    if exclude_failed
        xaxis = [[x for (x, s) in zip(xaxis, r) if s.success] for r in results]
        results = [[s for s in r if s.success] for r in results]
    end
    if take_avg
        Xs = vcat((mean(hcat((s.Xs for s in r)...), 1) for r in results)...)
        Xp = vcat((mean(hcat((s.Xp for s in r)...), 1) for r in results)...)
        s = vcat((mean(hcat((s.s for s in r)...), 1) for r in results)...)
        p = vcat((mean(hcat((s.p for s in r)...), 1) for r in results)...)
        payoffs = vcat((mean(hcat((s.payoffs for s in r)...), 1) for r in results)...)
        total_safety = [mean([s.σ for s in r]) for r in results]
        return xaxis, Xs, Xp, s, p, total_safety, payoffs
    else
        xaxis_ = vcat((fill(x, length(r)) for (x, r) in zip(xaxis, results))...)
        Xs = vcat((s.Xs for r in results for s in r)...)
        Xp = vcat((s.Xp for r in results for s in r)...)
        s = vcat((s.s for r in results for s in r)...)
        p = vcat((s.p for r in results for s in r)...)
        payoffs = vcat((s.payoffs for r in results for s in r)...)
        total_safety = [s.σ for r in results for s in r]
        return xaxis_, Xs, Xp, s, p, total_safety, payoffs        
    end
end

function create_scatterplots(results::Vector{Vector{SolverResult}}, xaxis, xlabel, labels; take_avg = false, exclude_failed = true)
    (xaxis_, Xs, Xp, s, p, total_safety, payoffs) = get_values_for_scatterplot(
        results, xaxis, take_avg = take_avg, exclude_failed = exclude_failed
    )
    labels_ = reshape(labels, 1, :)
    Xp_plt = scatter(xaxis_, Xp, xlabel = xlabel, ylabel = "Xₚ", labels = labels_)
    Xs_plt = scatter(xaxis_, Xs, xlabel = xlabel, ylabel = "Xₛ", labels = labels_)
    perf_plt = scatter(xaxis_, p, xlabel = xlabel, ylabel = "performance", labels = labels_)
    safety_plt = scatter(xaxis_, s, xlabel = xlabel, ylabel = "safety", labels = labels_)
    total_safety_plt = scatter(xaxis_, total_safety, xlabel = xlabel, ylabel = "σ", label = nothing)
    payoff_plt = scatter(xaxis_, payoffs, xlabel = xlabel, ylabel = "payoff", labels = labels_)
    return Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt
end

function create_scatterplot(
    results::Vector{Vector{SolverResult}}, xaxis, xlabel,
    plotsize, labels, title, logscale;
    take_avg = false, exclude_failed = true
)
    Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt = create_scatterplots(
        results, xaxis, xlabel, labels, take_avg = take_avg, exclude_failed = exclude_failed
    )
    if logscale
        yaxis!(Xp_plt, :log10)
        yaxis!(Xs_plt, :log10)
        yaxis!(perf_plt, :log10)
        yaxis!(safety_plt, :log10)
    end
    final_plot = plot(
        Xp_plt, Xs_plt,
        perf_plt, safety_plt,
        total_safety_plt, payoff_plt,
        layout = (3, 2), size = plotsize, legend_font_pointsize = 6,
        legend_background_color = RGBA(1., 1., 1., 0.5),
        markeralpha = 0.25, markerstrokealpha = 0.,
        left_margin = 20px
    )
    if !isnothing(title)
        plot!(plot_title = title)
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

function get_values_for_plot(results::Array{SolverResult, 2}; exclude_failed = true)
    (n_steps_secondary, n_steps) = size(results)
    n_players = size(results[1, 1].Xs, 1)
    Xs = fill(NaN, n_steps_secondary, n_steps, n_players)
    Xp = fill(NaN, n_steps_secondary, n_steps, n_players)
    s = fill(NaN, n_steps_secondary, n_steps, n_players)
    p = fill(NaN, n_steps_secondary, n_steps, n_players)
    payoffs = fill(NaN, n_steps_secondary, n_steps, n_players)
    total_safety = fill(NaN, n_steps_secondary, n_steps)
    for i in 1:n_steps_secondary, j in 1:n_steps
        if results[i, j].success || !exclude_failed
            Xs[i, j, :] = results[i, j].Xs
            Xp[i, j, :] = results[i, j].Xp
            s[i, j, :] = results[i, j].s
            p[i, j, :] = results[i, j].p
            payoffs[i, j, :] = results[i, j].payoffs
            total_safety[i, j] = results[i, j].σ
        end
    end
    return Xs, Xp, s, p, total_safety, payoffs
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


function _plot_helper_same(xaxis, Xs, Xp, s, p, total_safety, payoffs, xlabel, labels)
    n_steps_secondary = size(s)[1]
    colors = get_colors(n_steps_secondary)
    combine_values(x) = mean(x, 2)
    (Xp_plt, Xs_plt, perf_plt, safety_plt, payoff_plt) = (
        plot(
            xaxis, combine_values(x[1, :, :]),
            xlabel = xlabel, ylabel = ylab,
            labels = labels[1],
            color = colors[1]
        )
        for (x, ylab) in zip(
            (Xp, Xs, p, s, payoffs),
            ("Xₚ", "Xₛ", "performance", "safety", "payoff")
        )
    )
    total_safety_plt = plot(
        xaxis, total_safety[1, :],
        xlabel = xlabel, ylabel = "σ",
        label = labels[1],
        color = colors[1]
    )
    for i in 2:n_steps_secondary
        for (plt, x) in zip(
            (Xp_plt, Xs_plt, perf_plt, safety_plt, payoff_plt),
            (Xp, Xs, p, s, payoffs)
        )
            plot!(
                plt,
                xaxis, combine_values(x[i, :, :]),
                labels = labels[i],
                color = colors[i]
            )
        end
        plot!(
            total_safety_plt,
            xaxis, total_safety[i, :],
            label = labels[i],
            color = colors[i]
        )
    end
    return Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt
end

function _plot_helper_het(xaxis, Xs, Xp, s, p, total_safety, payoffs, xlabel, labels)
    (n_steps_secondary, _, n_players) = size(s)
    palettes = get_color_palettes(n_steps_secondary, n_players)
    colors = get_colors(n_steps_secondary)
    labels1 = reshape(["$(labels[1]), player $i" for i in 1:n_players], 1, :)
    (Xp_plt, Xs_plt, perf_plt, safety_plt, payoff_plt) = (
        plot(
            xaxis, x[1, :, :],
            xlabel = xlabel, ylabel = ylab,
            labels = labels1,
            palette = palettes[1]
        )
        for (x, ylab) in zip(
            (Xp, Xs, p, s, payoffs),
            ("Xₚ", "Xₛ", "performance", "safety", "payoff")
        )
    )
    total_safety_plt = plot(
        xaxis, total_safety[1, :],
        xlabel = xlabel, ylabel = "σ",
        label = labels[1],
        color = colors[1]
    )
    for i in 2:n_steps_secondary
        labelsi = reshape(["$(labels[i]), player $j" for j in 1:n_players], 1, :)
        for (plt, x) in zip(
            (Xp_plt, Xs_plt, perf_plt, safety_plt, payoff_plt),
            (Xp, Xs, p, s, payoffs)
        )
            plot!(
                plt,
                xaxis, x[i, :, :],
                labels = labelsi,
                palette = palettes[i]
            )
        end
        plot!(
            total_safety_plt,
            xaxis, total_safety[i, :],
            label = labels[i],
            color = colors[i]
        )
    end
    return Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt
end

function create_plots(results::Array{SolverResult, 2}, xaxis, xlabel, labels; exclude_failed = true)
    (Xs, Xp, s, p, total_safety, payoffs) = get_values_for_plot(results, exclude_failed = exclude_failed)
    players_same = are_players_same(results)
    return if players_same
        _plot_helper_same(xaxis, Xs, Xp, s, p, total_safety, payoffs, xlabel, labels)
    else
        _plot_helper_het(xaxis, Xs, Xp, s, p, total_safety, payoffs, xlabel, labels)
    end
end


function create_plot(results::Array{SolverResult, 2}, xaxis, xlabel, plotsize, labels, title, logscale; exclude_failed = true)
    (Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt) = create_plots(
        results, xaxis, xlabel, labels, exclude_failed = exclude_failed
    )
    if logscale
        yaxis!(Xp_plt, :log10)
        yaxis!(Xs_plt, :log10)
        yaxis!(perf_plt, :log10)
        yaxis!(safety_plt, :log10)
    end
    final_plot = plot(
        Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt,
        layout = (3, 2), size = plotsize, legend_font_pointsize = 6,
        legend_background_color = RGBA(1., 1., 1., 0.5),
        left_margin = 20px
    )
    if !isnothing(title)
        plot!(plot_title = title)
    end
    return final_plot
end

function get_xaxis_for_result_plot(res::ScenarioResult)
    varying_param = getfield(res.scenario, res.scenario.varying_param)
    n_steps = size(varying_param, 1)
    return if varying_param[:, 2] == varying_param[:, 1]
        varying_param[:, 1]
    else
        1:n_steps
    end
end

function get_labels_for_secondary_result_plot(res::ScenarioResult)
    secondary_varying_param = getfield(res.scenario, res.scenario.secondary_varying_param)
    n_steps_secondary = size(secondary_varying_param, 1)
    return ["$(res.scenario.secondary_varying_param) = $(secondary_varying_param[i, :])" for i in 1:n_steps_secondary]
end

function get_plots_for_result(
    res::ScenarioResult{SolverResult, 1};
    take_avg = false,
    exclude_failed = true
)
    xaxis = get_xaxis_for_result_plot(res)
    labels = ["player $i" for i in 1:res.scenario.n_players]
    create_plots(
        res.solverResults,
        xaxis,
        res.scenario.varying_param,
        labels,
        exclude_failed = exclude_failed
    )
end

function get_plots_for_result(
    res::ScenarioResult{SolverResult, 2};
    plotsize = (900, 900),
    title = nothing,
    logscale = false,
    take_avg = false,
    exclude_failed = true
)
    xaxis = get_xaxis_for_result_plot(res)
    labels = get_labels_for_secondary_result_plot(res)
    create_plots(
        res.solverResults,
        xaxis,
        res.scenario.varying_param,
        labels,
        exclude_failed = exclude_failed
    )
end

function get_plots_for_result(
    res::ScenarioResult{Vector{SolverResult}, 1};
    plotsize = (900, 900),
    title = nothing,
    logscale = false,
    take_avg = false,
    exclude_failed = true
)
    xaxis = get_xaxis_for_result_plot(res)
    labels = ["player $i" for i in 1:res.scenario.n_players]
    create_scatterplots(
        res.solverResults,
        xaxis,
        res.scenario.varying_param,
        labels,
        take_avg = take_avg,
        exclude_failed = exclude_failed
    )
end

function plot_result(
    res::ScenarioResult{SolverResult, 1};
    plotsize = (900, 900),
    title = nothing,
    logscale = false,
    take_avg = false,
    exclude_failed = true
)
    xaxis = get_xaxis_for_result_plot(res)
    labels = ["player $i" for i in 1:res.scenario.n_players]
    create_plot(
        res.solverResults,
        xaxis,
        res.scenario.varying_param,
        plotsize,
        labels,
        title,
        logscale,
        exclude_failed = exclude_failed
    )
end

function plot_result(
    res::ScenarioResult{SolverResult, 2};
    plotsize = (900, 900),
    title = nothing,
    logscale = false,
    take_avg = false,
    exclude_failed = true
)
    xaxis = get_xaxis_for_result_plot(res)
    labels = get_labels_for_secondary_result_plot(res)
    create_plot(
        res.solverResults,
        xaxis,
        res.scenario.varying_param,
        plotsize,
        labels,
        title,
        logscale,
        exclude_failed = exclude_failed
    )
end

function plot_result(
    res::ScenarioResult{Vector{SolverResult}, 1};
    plotsize = (900, 900),
    title = nothing,
    logscale = false,
    take_avg = false,
    exclude_failed = true
)
    xaxis = get_xaxis_for_result_plot(res)
    labels = ["player $i" for i in 1:res.scenario.n_players]
    create_scatterplot(
        res.solverResults,
        xaxis,
        res.scenario.varying_param,
        plotsize, 
        labels,
        title,
        logscale,
        take_avg = take_avg,
        exclude_failed = exclude_failed
    )
end


# FUNCTIONS FOR PLOTTING PAYOFFS (TO VISUALLY VERIFY EQUILIBRIUM)

function plot_payoffs_with_xs(
    problem, base_Xs, base_Xp, i;
    min_Xp = 0.001, max_Xp = 1., n_Xp = 40,
    logscale = false
)
    Xp = if logscale
        exp.(range(log(min_Xp), stop = log(max_Xp), length = n_Xp))
    else
        range(min_Xp, stop = max_Xp, length = n_Xp)
    end
    function f(xp)
        Xp_ = copy(vec(base_Xp))
        Xp_[i] = xp
        payoff(problem, i, vec(base_Xs), Xp_)
    end
    plot(Xp, f, xlabel = "Xₚ", ylabel = "payoff", legend = nothing)
    scatter!([base_Xp[i]], [f(base_Xp[i])])
    title!("Player $i")
end

function plot_payoffs_with_xp(
    problem, base_Xs, base_Xp, i;
    min_Xs = 0.001, max_Xs = 1., n_Xs = 40,
    logscale = false
)
    Xs = if logscale
        exp.(range(log(min_Xs), stop = log(max_Xs), length = n_Xs))
    else
        range(min_Xs, stop = max_Xs, length = n_Xs)
    end
    function f(xs)
        Xs_ = copy(vec(base_Xs))
        Xs_[i] = xs
        payoff(problem, i, Xs_, vec(base_Xp))
    end
    plot(Xs, f, xlabel = "Xₛ", ylabel = "payoff", legend = nothing)
    scatter!([base_Xs[i]], [f(base_Xs[i])])
    title!("Player $i")
end


function plot_payoffs(
    problem, base_Xs, base_Xp, i;
    min_Xs = 0.001, max_Xs = 1., n_Xs = 40,
    min_Xp = 0.001, max_Xp = 1., n_Xp = 40,
    logscale = false
)
    (Xs, Xp) = if logscale
        (
            exp.(range(log(min_Xs), stop = log(max_Xs), length = n_Xs)),
            exp.(range(log(min_Xp), stop = log(max_Xp), length = n_Xp))
        )
    else
        (
            range(min_Xs, stop = max_Xs, length = n_Xs),
            range(min_Xp, stop = max_Xp, length = n_Xp)
        )
    end
    function f(xs, xp)
        Xs_ = copy(vec(base_Xs)); Xp_ = copy(vec(base_Xp))
        Xs_[i] = xs; Xp_[i] = xp
        payoff(problem, i, Xs_, Xp_)
    end
    heatmap(Xs, Xp, f, xlabel = "Xₛ", ylabel = "Xₚ")
    scatter!([base_Xs[i]], [base_Xp[i]], legend = nothing)
    title!("Player $i")
end


function plot_payoffs_near_solution(problem, result::SolverResult)
    plot(
        plot_payoffs_with_xp(
            problem, result.Xs, result.Xp, 1,
            min_Xs = result.Xs[1] / 10, max_Xs = result.Xs[1] * 2 + EPSILON
        ),
        plot_payoffs_with_xs(
            problem, result.Xs, result.Xp, 1,
            min_Xp = result.Xp[1] / 10, max_Xp = result.Xp[1] * 2 + EPSILON
        ),
        plot_payoffs_with_xp(
            problem, result.Xs, result.Xp, 2,
            min_Xs = result.Xs[2] / 10, max_Xs = result.Xs[2] * 2 + EPSILON
        ),
        plot_payoffs_with_xs(
            problem, result.Xs, result.Xp, 2,
            min_Xp = result.Xp[2] / 10, max_Xp = result.Xp[2] * 2 + EPSILON
        ),
        layout = (2, 2)
    )
end

function plot_payoffs_near_solution(result::ScenarioResult, index::Integer)
    problem = get_problem_from_scenario(result.scenario, index)
    if isnothing(problem)
        return
    end
    plot_payoffs_near_solution(problem, result.solverResults[index])
end
