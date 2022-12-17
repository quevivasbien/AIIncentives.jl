# import dependencies and load module components

using StaticArrays
using Optim
using Plots
using Plots: RecipesBase.plot

using LinearAlgebra: diagind

include("utils.jl")
include("ProdFunc.jl")
include("RiskFunc.jl")
include("CSF.jl")
include("CostFunc.jl")
include("PayoffFunc.jl")
include("Problem.jl")
include("SolverResult.jl")
include("solve.jl")
include("scenarios.jl")
include("plotting.jl")
