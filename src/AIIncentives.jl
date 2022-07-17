module AIIncentives

using Optim, NLsolve
using Plots, Plots.PlotMeasures
# using DataFrames

export solve,
    
    ProdFunc,
    CSF,
    Problem,
    SolverOptions,
    Scenario,

    plot_result,

    f,
    payoff,
    all_payoffs,
    reward,
    all_rewards


include("ProdFunc.jl")
include("CSF.jl")
include("Problem.jl")
include("SolverResult.jl")
include("solve.jl")
include("scenarios.jl")
# include("make_grid.jl")

end
