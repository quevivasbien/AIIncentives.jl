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
    Problem,
    SolverOptions,
    Scenario,

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


include("utils.jl")
include("ProdFunc.jl")
include("RiskFunc.jl")
include("CSF.jl")
include("PayoffFunc.jl")
include("Problem.jl")
include("SolverResult.jl")
include("solve.jl")
include("scenarios.jl")
include("plotting.jl")

end
