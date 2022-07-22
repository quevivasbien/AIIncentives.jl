module AIIncentives

using Optim, NLsolve
using Plots, Plots.PlotMeasures
# using DataFrames

export
    solve,
    
    ProdFunc,
    CSF,
    Problem,
    SolverOptions,
    Scenario,

    plot_result,
    plot_payoffs_with_xs,
    plot_payoffs_with_xp,
    plot_payoffs,
    plot_payoffs_near_solution,

    f,
    payoff,
    all_payoffs,
    reward,
    all_rewards


include("utils.jl")
include("ProdFunc.jl")
include("CSF.jl")
include("Problem.jl")
include("SolverResult.jl")
include("solve.jl")
include("scenarios.jl")
include("plotting.jl")
# include("make_grid.jl")

end
