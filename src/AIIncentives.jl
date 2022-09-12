module AIIncentives

using Optim
using Plots, Plots.PlotMeasures
using Plots: RecipesBase.plot

using LinearAlgebra: diagind

export
    solve,
    
    ProdFunc,
    MultiplicativeRisk,
    AdditiveRisk,
    WinnerOnlyRisk,
    LinearPayoff,
    PayoffOnDisaster,
    Problem,
    ProblemWithBeliefs,
    SolverOptions,
    SolverResult,
    Scenario,
    ScenarioResult,

    get_plots,
    get_plots_for_result,  # deprecated
    plot,
    plot_result,  # deprecated
    plot_payoffs_with_xs,
    plot_payoffs_with_xp,
    plot_payoffs,
    plot_payoffs_near_solution,

    f,
    payoff,
    payoffs,
    reward,
    all_rewards


include("includes.jl")

end
