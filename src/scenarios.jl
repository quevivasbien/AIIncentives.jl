mutable struct Scenario
    n_players::Integer
    A::Array
    α::Array
    B::Array
    β::Array
    θ::Array
    d::Array
    r::Array
    csf::CSF
    varying_param::Symbol
    secondary_varying_param::Union{Symbol, Nothing}
end

function asarray(x::Union{Real, AbstractArray}, n::Integer)
    if isa(x, AbstractArray)
        return x
    end
    return fill(x, n)
end

function check_param_sizes(scenario::Scenario)
    all(
        (
            x == scenario.varying_param
            || x == scenario.secondary_varying_param
            || size(getfield(scenario, x), 1) == scenario.n_players
        )
        for x in [:A, :α, :B, :β, :θ, :d, :r]
    )
end

function Scenario(
    ;
    n_players::Integer = 2,
    A::Union{Real, AbstractArray} = 10., α::Union{Real, AbstractArray} = 0.5,
    B::Union{Real, AbstractArray} = 10., β::Union{Real, AbstractArray} = 0.5,
    θ::Union{Real, AbstractArray} = 0.5,
    d::Union{Real, AbstractArray} = 0.,
    r::Union{Real, AbstractArray} = 0.01:0.01:0.1,
    w::Real = 1., l::Real = 0., a_w::Real = 0., a_l::Real = 0.,
    varying_param::Symbol = :r,
    secondary_varying_param::Union{Symbol, Nothing} = nothing
)
    scenario = Scenario(
        n_players,
        asarray(A, n_players), asarray(α, n_players),
        asarray(B, n_players), asarray(β, n_players),
        asarray(θ, n_players),
        asarray(d, n_players), asarray(r, n_players), CSF(w, l, a_w, a_l),
        varying_param, secondary_varying_param
    )
    @assert check_param_sizes(scenario) "Your input params need to match the number of players"
    # expand varying params if necessary
    vparam = getfield(scenario, varying_param)
    if size(vparam, 2) == 1
        setfield!(
            scenario, varying_param,
            repeat(vparam, 1, n_players)
        )
    end
    if !isnothing(secondary_varying_param)
        vparam2 = getfield(scenario, secondary_varying_param)
        if size(vparam2, 2) == 1
            setfield!(
                scenario, secondary_varying_param,
                repeat(vparam2, 1, n_players)
            )
        end
    end
    return scenario
end


struct ScenarioResult
    scenario::Scenario
    solverResults::Array{SolverResult}
end

function extract(res::ScenarioResult, field::Symbol)
    if field in (:success, :Xs, :Xp, :s, :p, :payoffs)
        [getfield(x, field) for x in res.solverResults]
    else
        getfield(res.scenario, field)
    end
end

function get_problem_from_scenario(scenario::Scenario, index)
    if index isa Integer
        @assert isnothing(scenario.secondary_varying_param)
    end
    (_, A, α, B, β, θ, d, r) = if isnothing(scenario.secondary_varying_param)
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
        scenario.csf
    )
end


# SOLVER FUNCTIONS:

function get_params_with_secondary_variation(scenario)
    varying_param = transpose(getfield(scenario, scenario.varying_param))
    n_steps = size(varying_param, 2)
    secondary_varying_param = transpose(getfield(scenario, scenario.secondary_varying_param))
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
    return (n_steps, n_steps_secondary), A, α, B, β, θ, d, r
end

function solve_with_secondary_variation(
    scenario::Scenario,
    method::Symbol,
    options
)
    if method == :scatter || method == :mixed
        println("Secondary variation is unsupported with the mixed or scatter solvers.")
        return ScenarioResult(scenario, SolverResult[])
    end
    
    ((n_steps, n_steps_secondary), A, α, B, β, θ, d, r) = get_params_with_secondary_variation(scenario)

    results = Array{SolverResult}(undef, (n_steps_secondary, n_steps))
    Threads.@threads for i in 1:n_steps
        Threads.@threads for j in 1:n_steps_secondary
            prodFunc = ProdFunc(A[:, i, j], α[:, i, j], B[:, i, j], β[:, i, j], θ[:, i, j])
            problem = Problem(d[:, i, j], r[:, i, j],  prodFunc, scenario.csf)
            results[j, i] = solve(problem, method::Symbol, options)
        end
    end
    if options.verbose
        print.(results)
    end

    return ScenarioResult(scenario, results)
end

function get_params(scenario)
    varying_param = transpose(getfield(scenario, scenario.varying_param))
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
    return n_steps, A, α, B, β, θ, d, r
end

function solve(scenario::Scenario, method::Symbol, options)
    if !isnothing(scenario.secondary_varying_param)
        return solve_with_secondary_variation(
            scenario,
            method::Symbol,
            options
        )
    end

    (n_steps, A, α, B, β, θ, d, r) = get_params(scenario)

    results = Array{SolverResult}(undef, n_steps)
    # send to solver
    Threads.@threads for i in 1:n_steps
        prodFunc = ProdFunc(A[:, i], α[:, i], B[:, i], β[:, i], θ[:, i])
        problem = Problem(d[:, i], r[:, i],  prodFunc, scenario.csf)
        results[i] = solve(problem, method::Symbol, options)
    end

    return ScenarioResult(scenario, results)
end

function solve(scenario::Scenario; method::Symbol = :iters, kwargs...)
    options = SolverOptions(SolverOptions(); kwargs...)
    return solve(scenario, method::Symbol, options)
end


function test_scenarios(method = :hybrid)
    println("Running test on `scenarios.jl`...")

    scenario = Scenario(
        n_players = 2,
        α = [0.5, 0.75],
        θ = [0., 0.5],
        r = range(0.01, 0.1, length = Threads.nthreads()),
        secondary_varying_param = :θ
    )

    @time res = solve(scenario, method = method, verbose  = true)
    plot_result(res)
end
