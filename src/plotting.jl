# Functions for plotting with only single varying param

function get_values_for_plot(results::Vector{SolverResult}, exclude_failed = true)
    n_steps = length(results)
    n = size(results[1].Xs, 1)
    Xs = fill(NaN, n_steps, n)
    Xp = fill(NaN, n_steps, n)
    s = fill(NaN, n_steps, n)
    p = fill(NaN, n_steps, n)
    payoffs = fill(NaN, n_steps, n)
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

function create_plots(results::Vector{SolverResult}, xvals, xlabel, labels, exclude_failed = true; kwargs...)
    (Xs, Xp, s, p, total_safety, payoffs) = get_values_for_plot(results, exclude_failed)
    labels_ = reshape(labels, 1, :)
    Xp_plt = plot(xvals, Xp, xlabel = xlabel, ylabel = "Xₚ", labels = labels_; kwargs...)
    Xs_plt = plot(xvals, Xs, xlabel = xlabel, ylabel = "Xₛ", labels = labels_; kwargs...)
    perf_plt = plot(xvals, p, xlabel = xlabel, ylabel = "performance", labels = labels_; kwargs...)
    safety_plt = plot(xvals, s, xlabel = xlabel, ylabel = "safety", labels = labels_; kwargs...)
    total_safety_plt = plot(xvals, total_safety, xlabel = xlabel, ylabel = "σ", label = nothing; kwargs...)
    payoff_plt = plot(xvals, payoffs, xlabel = xlabel, ylabel = "payoff", labels = labels_; kwargs...)
    return Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt
end

function create_plot(results::Vector{SolverResult}, xvals, xlabel, labels, logscale, exclude_failed = true; kwargs...)
    Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt = create_plots(results, xvals, xlabel, labels, exclude_failed; kwargs...)
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
        layout = (3, 2), size = (900, 900), legend_font_pointsize = 6,
        legend_background_color = RGBA(1., 1., 1., 0.5),
        left_margin = 20px
    )
    return final_plot
end

# functions for plotting with single varying param, scatterplot instead of line plot (for when multiple solutions are found)

function get_values_for_scatterplot(results::Vector{Vector{SolverResult}}, xvals, take_avg = false, exclude_failed = true)
    if exclude_failed
        xvals = [fill(x, sum((r.success for r in s))) for (x, s) in zip(xvals, results)]
        results = [[s for s in r if s.success] for r in results]
    else
        xvals = [fill(x, length(s)) for (x, s) in zip(xvals, results)]
    end
    if take_avg
        xvals_ = [mean(x) for x in xvals]
        Xs = hcat((mean(hcat((s.Xs for s in r)...), 2) for r in results)...) |> transpose
        Xp = hcat((mean(hcat((s.Xp for s in r)...), 2) for r in results)...) |> transpose
        s = hcat((mean(hcat((s.s for s in r)...), 2) for r in results)...) |> transpose
        p = hcat((mean(hcat((s.p for s in r)...), 2) for r in results)...) |> transpose
        payoffs = hcat((mean(hcat((s.payoffs for s in r)...), 2) for r in results)...) |> transpose
        total_safety = [mean([s.σ for s in r]) for r in results]
        return xvals_, Xs, Xp, s, p, total_safety, payoffs
    else
        xvals_ = vcat(xvals...)
        Xs = hcat((s.Xs for r in results for s in r)...) |> transpose
        Xp = hcat((s.Xp for r in results for s in r)...) |> transpose
        s = hcat((s.s for r in results for s in r)...) |> transpose
        p = hcat((s.p for r in results for s in r)...) |> transpose
        payoffs = hcat((s.payoffs for r in results for s in r)...) |> transpose
        total_safety = [s.σ for r in results for s in r]
        return xvals_, Xs, Xp, s, p, total_safety, payoffs        
    end
end

function create_scatterplots(results::Vector{Vector{SolverResult}}, xvals, xlabel, labels, take_avg = false, exclude_failed = true; kwargs...)
    (xvals_, Xs, Xp, s, p, total_safety, payoffs) = get_values_for_scatterplot(
        results, xvals, take_avg, exclude_failed
    )
    labels_ = reshape(labels, 1, :)
    Xp_plt = scatter(xvals_, Xp, xlabel = xlabel, ylabel = "Xₚ", labels = labels_; kwargs...)
    Xs_plt = scatter(xvals_, Xs, xlabel = xlabel, ylabel = "Xₛ", labels = labels_; kwargs...)
    perf_plt = scatter(xvals_, p, xlabel = xlabel, ylabel = "performance", labels = labels_; kwargs...)
    safety_plt = scatter(xvals_, s, xlabel = xlabel, ylabel = "safety", labels = labels_; kwargs...)
    total_safety_plt = scatter(xvals_, total_safety, xlabel = xlabel, ylabel = "σ", label = nothing; kwargs...)
    payoff_plt = scatter(xvals_, payoffs, xlabel = xlabel, ylabel = "payoff", labels = labels_; kwargs...)
    return Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt
end

function create_scatterplot(
    results::Vector{Vector{SolverResult}}, xvals, xlabel,
    labels, logscale,
    take_avg = false, exclude_failed = true;
    kwargs...
)
    Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt = create_scatterplots(
        results, xvals, xlabel, labels, take_avg, exclude_failed; kwargs...
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
        layout = (3, 2), size = (900, 900), legend_font_pointsize = 6,
        legend_background_color = RGBA(1., 1., 1., 0.5),
        markeralpha = 0.25, markerstrokealpha = 0.,
        left_margin = 20px
    )
    return final_plot
end


# Functions for plotting with secondary variation

# kind of a garish color scheme, but meant to be suitable when printed in b&w
function get_colors(n_lines)
    return HSV.(range(240, step = 25, length = n_lines), 1., 1.) |> to_rowvec
end

function get_color_palettes(n_lines, n)
    return [
        palette(
            HSV.(hue, range(1., 0.4, length = n), range(0.5, 1., length = n))
        ) for hue in range(240, step = 25, length = n_lines)
    ] 
end

function get_values_for_plot(results::Array{SolverResult, 2}, exclude_failed = true)
    (n_steps, n_steps_secondary) = size(results)
    n = size(results[1, 1].Xs, 1)
    Xs = fill(NaN, n_steps, n_steps_secondary, n)
    Xp = fill(NaN, n_steps, n_steps_secondary, n)
    s = fill(NaN, n_steps, n_steps_secondary, n)
    p = fill(NaN, n_steps, n_steps_secondary, n)
    payoffs = fill(NaN, n_steps, n_steps_secondary, n)
    total_safety = fill(NaN, n_steps, n_steps_secondary)
    for i in 1:n_steps, j in 1:n_steps_secondary
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
    (n_steps, n_steps_secondary) = size(results)
    n = size(results[1, 1].Xs, 1)
    strats = Array{Float64}(undef, n_steps, n_steps_secondary, n, 2)
    for i in 1:n_steps, j in 1:n_steps_secondary
        strats[i, j, :, 1] = results[i, j].Xs
        strats[i, j, :, 2] = results[i, j].Xp
    end
    return all(
        isapprox(
            strats[:, :, 1, :] - strats[:, :, i, :],
            zeros(n_steps, n_steps_secondary, 2),
            atol=1e-2
        ) for i in 2:n
    )
end


function _plot_helper_same(xvals, Xs, Xp, s, p, total_safety, payoffs, xlabel, labels; kwargs...)
    n_steps_secondary = size(s, 2)
    colors = get_colors(n_steps_secondary)
    (Xp_plt, Xs_plt, perf_plt, safety_plt, payoff_plt) = (
        plot(
            xvals, mean(x, 3),
            xlabel = xlabel, ylabel = ylab,
            labels = labels,
            colors = colors;
            kwargs...
        )
        for (x, ylab) in zip(
            (Xp, Xs, p, s, payoffs),
            ("Xₚ", "Xₛ", "performance", "safety", "payoff")
        )
    )
    total_safety_plt = plot(
        xvals, total_safety,
        xlabel = xlabel, ylabel = "σ",
        labels = labels,
        colors = colors;
        kwargs...
    )
    return Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt
end

function _plot_helper_het(xvals, Xs, Xp, s, p, total_safety, payoffs, xlabel, labels; kwargs...)
    (_, n_steps_secondary, n) = size(s)
    palettes = get_color_palettes(n_steps_secondary, n)
    colors = get_colors(n_steps_secondary)
    (Xp_plt, Xs_plt, perf_plt, safety_plt, payoff_plt) = (
        plot(
            xlabel = xlabel, ylabel = ylab;
            kwargs...
        )
        for ylab in ("Xₚ", "Xₛ", "performance", "safety", "payoff")
    )
    total_safety_plt = plot(
        xvals, total_safety,
        xlabel = xlabel, ylabel = "σ",
        labels = labels,
        colors = colors;
        kwargs...
    )
    for i in 1:n_steps_secondary
        labelsi = reshape(["$(labels[i]), player $j" for j in 1:n], 1, :)
        for (plt, x) in zip(
            (Xp_plt, Xs_plt, perf_plt, safety_plt, payoff_plt),
            (Xp, Xs, p, s, payoffs)
        )
            plot!(
                plt,
                xvals, x[:, i, :],
                labels = labelsi,
                palette = palettes[i]
            )
        end
    end
    return Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt
end

function create_plots(results::Array{SolverResult, 2}, xvals, xlabel, labels, exclude_failed = true; kwargs...)
    (Xs, Xp, s, p, total_safety, payoffs) = get_values_for_plot(results, exclude_failed)
    players_same = are_players_same(results)
    return if players_same
        _plot_helper_same(xvals, Xs, Xp, s, p, total_safety, payoffs, xlabel, labels; kwargs...)
    else
        _plot_helper_het(xvals, Xs, Xp, s, p, total_safety, payoffs, xlabel, labels; kwargs...)
    end
end


function create_plot(results::Array{SolverResult, 2}, xvals, xlabel, labels, logscale, exclude_failed = true; kwargs...)
    (Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt) = create_plots(
        results, xvals, xlabel, labels, exclude_failed; kwargs...
    )
    if logscale
        yaxis!(Xp_plt, :log10)
        yaxis!(Xs_plt, :log10)
        yaxis!(perf_plt, :log10)
        yaxis!(safety_plt, :log10)
    end
    final_plot = plot(
        Xp_plt, Xs_plt, perf_plt, safety_plt, total_safety_plt, payoff_plt,
        layout = (3, 2), size = (900, 900), legend_font_pointsize = 6,
        legend_background_color = RGBA(1., 1., 1., 0.5),
        left_margin = 20px
    )
    return final_plot
end

function get_xvals_for_result_plot(res::ScenarioResult)
    varying = reduce(vcat, transpose.(extract(res.scenario, res.scenario.varying)[:, 1]))
    return if varying[:, 2] == varying[:, 1]
        varying[:, 1]
    else
        1:scenario.n_steps
    end
end

function get_labels_for_plot(res::ScenarioResult, labels = nothing)
    if !isnothing(labels)
        if typeof(labels) <: AbstractVector
            to_rowvec(labels)
        else
            labels
        end
    elseif isnothing(res.scenario.varying2)
        ["player $i" for i in 1:res.scenario.n] |> to_rowvec
    else
        varying2 = reduce(
            vcat,
            transpose.(extract(res.scenario, res.scenario.varying2)[1, :])
        )
        if varying2[:, 2] == varying2[:, 1]
            ["$(res.scenario.varying2) = $v" for v in varying2[:, 1]]
        else
            ["$(res.scenario.varying2) = $v" for v in rows(varying2)]
        end |> to_rowvec
    end
end

# for results with no secondary variation and only one result per problem
function get_plots(
    res::ScenarioResult{SolverResult, 1};
    xvals = nothing,
    labels = nothing,
    take_avg = false,
    exclude_failed = true,
    kwargs...
)
    xvals = isnothing(xvals) ? get_xvals_for_result_plot(res) : xvals
    labels = get_labels_for_plot(res, labels)
    create_plots(
        res.solverResults,
        xvals,
        res.scenario.varying,
        labels, 
        exclude_failed;
        kwargs...
    )
end

# for scenarios with secondary variation
function get_plots(
    res::ScenarioResult{SolverResult, 2};
    xvals = nothing,
    labels = nothing,
    take_avg = false,
    exclude_failed = true,
    kwargs...
)
    xvals = isnothing(xvals) ? get_xvals_for_result_plot(res) : xvals
    labels = get_labels_for_plot(res, labels)
    create_plots(
        res.solverResults,
        xvals,
        res.scenario.varying,
        labels,
        exclude_failed;
        kwargs...
    )
end

# for results with no secondary variation and multiple results per problem
function get_plots(
    res::ScenarioResult{Vector{SolverResult}, 1};
    xvals = nothing,
    labels = nothing,
    take_avg = false,
    exclude_failed = true,
    kwargs...
)
    xvals = isnothing(xvals) ? get_xvals_for_result_plot(res) : xvals
    labels = get_labels_for_plot(res, labels)
    create_scatterplots(
        res.solverResults,
        xvals,
        res.scenario.varying,
        labels,
        take_avg,
        exclude_failed;
        kwargs...
    )
end

# for results with no secondary variation and only one result per problem
function RecipesBase.plot(
    res::ScenarioResult{SolverResult, 1};
    xvals = nothing,
    labels = nothing,
    logscale = false,
    take_avg = false,
    exclude_failed = true,
    kwargs...
)
    xvals = isnothing(xvals) ? get_xvals_for_result_plot(res) : xvals
    labels = get_labels_for_plot(res, labels)
    create_plot(
        res.solverResults,
        xvals,
        res.scenario.varying,
        labels,
        logscale,
        exclude_failed;
        kwargs...
    )
end

function get_plots_for_result(res; kwargs...)
    return get_plots(res; kwargs...)
end
@deprecate get_plots_for_result get_plots

# for scenarios with secondary variation
function RecipesBase.plot(
    res::ScenarioResult{SolverResult, 2};
    xvals = nothing,
    labels = nothing,
    logscale = false,
    take_avg = false,
    exclude_failed = true,
    kwargs...
)
    xvals = isnothing(xvals) ? get_xvals_for_result_plot(res) : xvals
    labels = get_labels_for_plot(res, labels)
    create_plot(
        res.solverResults,
        xvals,
        res.scenario.varying,
        labels,
        logscale,
        exclude_failed;
        kwargs...
    )
end

# for results with no secondary variation and multiple results per problem
function RecipesBase.plot(
    res::ScenarioResult{Vector{SolverResult}, 1};
    xvals = nothing,
    labels = nothing,
    logscale = false,
    take_avg = false,
    exclude_failed = true,
    kwargs...
)
    xvals = isnothing(xvals) ? get_xvals_for_result_plot(res) : xvals
    labels = get_labels_for_plot(res, labels)
    create_scatterplot(
        res.solverResults,
        xvals,
        res.scenario.varying,
        labels,
        logscale,
        take_avg,
        exclude_failed;
        kwargs...
    )
end

function plot_result(res; kwargs...)
    return plot(res; kwargs...)
end
@deprecate plot_result plot

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
        get_payoff(problem, i, vec(base_Xs), Xp_)
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
        get_payoff(problem, i, Xs_, vec(base_Xp))
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
        get_payoff(problem, i, Xs_, Xp_)
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

function plot_payoffs_near_solution(result::ScenarioResult, index::Int)
    problem = get_problem_from_scenario(result.scenario, index)
    if isnothing(problem)
        return
    end
    plot_payoffs_near_solution(problem, result.solverResults[index])
end
