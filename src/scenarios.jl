mutable struct Scenario{T}
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

function get_problem_from_scenario(scenario::Scenario, index)
    if index isa Int
        @assert isnothing(scenario.varying2)
    end
    (_, A, α, B, β, θ, d, r) = if isnothing(scenario.varying2)
        get_params(scenario)
    else
        get_params_with_secondary_variation(scenario)
    end

    Problem(
        scenario.n_players,
        d[:, index...], r[:, index...],
        ProdFunc(
            scenario.n_players,
            A[:, index...], α[:, index...],
            B[:, index...], β[:, index...],
            θ[:, index...]
        ),
        scenario.riskFunc,
        scenario.csf,
        scenario.payoffFunc
    )
end


# SOLVER FUNCTIONS:

function get_params_with_secondary_variation(scenario)
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
    return (n_steps, n_steps_secondary), A, α, B, β, θ, d, r
end

function solve_with_secondary_variation(
    scenario::Scenario,
    method::Symbol,
    options
)
    @assert method != :scatter && method != :mixed "Secondary variation is unsupported with the mixed or scatter solvers."
    
    ((n_steps, n_steps_secondary), A, α, B, β, θ, d, r) = get_params_with_secondary_variation(scenario)

    results = if method in (:scatter, :mixed)
        Array{Vector{SolverResult}}(undef, (n_steps_secondary, n_steps))
    else
        Array{SolverResult}(undef, (n_steps_secondary, n_steps))
    end
    Threads.@threads for i in 1:n_steps
        Threads.@threads for j in 1:n_steps_secondary
            problem = get_problem_from_scenario(scenario, (i, j))
            results[j, i] = solve(problem, method, options)
        end
    end
    if options.verbose
        print.(results)
    end

    return ScenarioResult(scenario, results)
end

function get_params(scenario)
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
    return n_steps, A, α, B, β, θ, d, r
end

function solve(scenario::Scenario, method::Symbol, options)
    if !isnothing(scenario.varying2)
        return solve_with_secondary_variation(
            scenario,
            method,
            options
        )
    end

    (n_steps, A, α, B, β, θ, d, r) = get_params(scenario)

    results = if method in (:mixed, :scatter)
        Array{Vector{SolverResult}}(undef, n_steps)
    else
        Array{SolverResult}(undef, n_steps)
    end
    # send to solver
    Threads.@threads for i in 1:n_steps
        prodFunc = ProdFunc(A[:, i], α[:, i], B[:, i], β[:, i], θ[:, i])
        problem = get_problem_from_scenario(scenario, i)
        results[i] = solve(problem, method, options)
    end

    return ScenarioResult(scenario, results)
end

function solve(scenario::Scenario; method::Symbol = :iters, kwargs...)
    options = SolverOptions(SolverOptions(); kwargs...)
    return solve(scenario, method::Symbol, options)
end
