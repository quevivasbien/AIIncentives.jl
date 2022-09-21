abstract type AbstractScenario end

mutable struct Scenario{T} <: AbstractScenario
    n_players::Int
    A::Array{T}
    α::Array{T}
    B::Array{T}
    β::Array{T}
    θ::Array{T}
    d::Array{T}
    r::Array{T}
    riskFunc::RiskFunc
    csf::CSF
    payoffFunc::PayoffFunc
    varying::Symbol
    varying2::Union{Symbol, Nothing}
end

function check_param_sizes(scenario::Scenario)
    all(
        (
            x == scenario.varying
            || x == scenario.varying2
            || size(getfield(scenario, x), 1) == scenario.n_players
        )
        for x in [:A, :α, :B, :β, :θ, :d, :r]
    )
end

function Scenario(
    ;
    n_players::Int = 2,
    A::Union{Real, AbstractArray} = 10., α::Union{Real, AbstractArray} = 0.5,
    B::Union{Real, AbstractArray} = 10., β::Union{Real, AbstractArray} = 0.5,
    θ::Union{Real, AbstractArray} = 0.5,
    d::Union{Real, AbstractArray} = 0.,
    r::Union{Real, AbstractArray} = 0.01:0.01:0.1,
    riskFunc::RiskFunc = WinnerOnlyRisk(),
    csf::CSF = BasicCSF(),
    payoffFunc::PayoffFunc = LinearPayoff(),
    varying::Symbol = :r,
    varying2::Union{Symbol, Nothing} = nothing,
    varying_param = nothing,
    secondary_varying_param = nothing
)
    @assert isnothing(varying_param) && isnothing(secondary_varying_param) "Use `varying` and `varying2` in place of `varying_param` and `secondary_varying_param`, respectively"
    scenario = Scenario(
        n_players,
        as_Float64_Array(A, n_players), as_Float64_Array(α, n_players),
        as_Float64_Array(B, n_players), as_Float64_Array(β, n_players),
        as_Float64_Array(θ, n_players),
        as_Float64_Array(d, n_players), as_Float64_Array(r, n_players),
        riskFunc, csf, payoffFunc,
        varying, varying2
    )
    @assert check_param_sizes(scenario) "Your input params need to match the number of players"
    # expand varying params if necessary
    vparam = getfield(scenario, varying)
    if size(vparam, 2) == 1
        setfield!(
            scenario, varying,
            repeat(vparam, 1, n_players)
        )
    end
    if !isnothing(varying2)
        vparam2 = getfield(scenario, varying2)
        if size(vparam2, 2) == 1
            setfield!(
                scenario, varying2,
                repeat(vparam2, 1, n_players)
            )
        end
    end
    return scenario
end


struct ScenarioResult{T <: Union{SolverResult, Vector{SolverResult}}, N}
    scenario::Scenario
    solverResults::Array{T, N}
end


function extract(res::ScenarioResult{SolverResult, 1}, field::Symbol)
    if field in (:success, :Xs, :Xp, :s, :p, :σ, :payoffs)
        [getfield(x, field) for x in res.solverResults]
    else
        getfield(res.scenario, field)
    end
end

function get_params(scenario::Scenario)
    varying = transpose(getfield(scenario, scenario.varying))
    n_steps = size(varying, 2)
    
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
        if symbol == scenario.varying
            copy!(newvar, varying)
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
    return A, α, B, β, θ, d, r
end

function get_params_with_secondary_variation(scenario::Scenario)
    varying = transpose(getfield(scenario, scenario.varying))
    n_steps = size(varying, 2)
    varying2 = transpose(getfield(scenario, scenario.varying2))
    n_steps_secondary = size(varying2, 2)
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
        if symbol == scenario.varying
            copyto!(
                newvar,
                repeat(varying, outer = (1, n_steps_secondary))
            )
        elseif symbol == scenario.varying2
            copyto!(
                newvar,
                repeat(varying2, inner = (1, n_steps))
            )
        else
            var = getfield(scenario, symbol)
            copyto!(
                newvar,
                repeat(var, inner = (1, n_steps), outer = (1, n_steps_secondary))
            )
        end
    end
    return A, α, B, β, θ, d, r
end

function get_problem_from_scenario(scenario::Scenario, index)
    idx = Tuple(index)
    if length(idx) == 1
        @assert isnothing(scenario.varying2)
    end
    (A, α, B, β, θ, d, r) = if isnothing(scenario.varying2)
        get_params(scenario)
    else
        get_params_with_secondary_variation(scenario)
    end

    return Problem(
        scenario.n_players,
        d[:, idx...], r[:, idx...],
        ProdFunc(
            scenario.n_players,
            A[:, idx...], α[:, idx...],
            B[:, idx...], β[:, idx...],
            θ[:, idx...]
        ),
        scenario.riskFunc,
        scenario.csf,
        scenario.payoffFunc
    )
end


# SOLVER FUNCTIONS:

function setup_results(scenario::Scenario, method)
    if !isnothing(scenario.varying2)
        @assert method != :scatter && method != :mixed "Secondary variation is unsupported with the mixed or scatter solvers."

        n_steps = size(getfield(scenario, scenario.varying), 1)
        n_steps_secondary = size(getfield(scenario, scenario.varying2), 1)
        return if method in (:scatter, :mixed)
            Array{Vector{SolverResult}}(undef, (n_steps_secondary, n_steps))
        else
            Array{SolverResult}(undef, (n_steps_secondary, n_steps))
        end
    else
        n_steps = size(getfield(scenario, scenario.varying), 1)
        return if method in (:mixed, :scatter)
            Array{Vector{SolverResult}}(undef, n_steps)
        else
            Array{SolverResult}(undef, n_steps)
        end
    end
end

function fill_results!(results, scenario, method, options)
    indices = CartesianIndices(size(results))
    Threads.@threads for index in indices
        problem = get_problem_from_scenario(scenario, (reverse ∘ Tuple)(index))
        results[index] = solve(problem, method, options)
    end
end

function solve(scenario::AbstractScenario, method, options)
    
    # create an empty array of the correct size to contain the results
    results = setup_results(scenario, method)
    # send to solver
    fill_results!(results, scenario, method, options)

    return ScenarioResult(scenario, results)
end

function solve(scenario::AbstractScenario; method = :iters, kwargs...)
    options = SolverOptions(SolverOptions(); kwargs...)
    return solve(scenario, method, options)
end


"""
Variation of scenario that encapsulates ProblemWithBeliefs
Provide a dict of idiosyncratic beliefs for each player

baseScenario determines baseProblem and how problem varies
beliefs determine how each player's belief is different from baseProblem
"""
struct ScenarioWithBeliefs{T} <: AbstractScenario
    baseScenario::Scenario{T}
    beliefs::Vector{Dict{Symbol, Vector{T}}}
end

function ScenarioWithBeliefs(
    baseScenario = Scenario();
    beliefs = fill(Dict(), Scenario().n_players)
)
    n = baseScenario.n_players
    @assert length(beliefs) == n
    # expand beliefs so values are vectors of length n
    beliefs_ = [
        Dict{Symbol, Vector{Float64}}(
            k => as_Float64_Array(v, n) for (k, v) in b
        )
        for b in beliefs
    ]
    @assert all(length(x) == n for b in beliefs_ for x in values(b))
    return ScenarioWithBeliefs(baseScenario, beliefs_)
end

function replace_belief(baseObj, field::Symbol, belief::Dict)
    return if field in keys(belief)
        belief[field]
    else
        getfield(baseObj, field)
    end
end

function setup_results(scenario::ScenarioWithBeliefs, method)
    return setup_results(scenario.baseScenario, method)
end

function get_problem_from_scenario(scenario::ScenarioWithBeliefs, index)
    idx = Tuple(index)
    if length(idx) == 1
        @assert isnothing(scenario.baseScenario.varying2)
    end
    (A, α, B, β, θ, d, r) = if isnothing(scenario.baseScenario.varying2)
        get_params(scenario.baseScenario)
    else
        get_params_with_secondary_variation(scenario.baseScenario)
    end

    baseProblem = Problem(
        scenario.baseScenario.n_players,
        d[:, idx...], r[:, idx...],
        ProdFunc(
            scenario.baseScenario.n_players,
            A[:, idx...], α[:, idx...],
            B[:, idx...], β[:, idx...],
            θ[:, idx...]
        ),
        scenario.baseScenario.riskFunc,
        scenario.baseScenario.csf,
        scenario.baseScenario.payoffFunc
    )

    beliefs = [
        Problem(
            scenario.baseScenario.n_players,
            replace_belief(baseProblem, :d, belief),
            replace_belief(baseProblem, :r, belief),
            ProdFunc(
                scenario.baseScenario.n_players,
                replace_belief(baseProblem.prodFunc, :A, belief),
                replace_belief(baseProblem.prodFunc, :α, belief),
                replace_belief(baseProblem.prodFunc, :B, belief),
                replace_belief(baseProblem.prodFunc, :β, belief),
                replace_belief(baseProblem.prodFunc, :θ, belief)
            ),
            scenario.baseScenario.riskFunc,
            scenario.baseScenario.csf,
            scenario.baseScenario.payoffFunc
        )
        for belief in scenario.beliefs
    ]

    return ProblemWithBeliefs(baseProblem, beliefs = beliefs)
end

function ScenarioResult(scenario::ScenarioWithBeliefs, results)
    return ScenarioResult(scenario.baseScenario, results)
end
