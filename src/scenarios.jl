abstract type AbstractScenario end

struct Scenario{T <: Problem, N} <: AbstractScenario
    n::Int
    n_steps::Int
    n_steps2::Union{Int, Nothing}
    varying::Symbol
    varying2::Union{Symbol, Nothing}
    varying_data::Array
    varying2_data::Union{Array, Nothing}
    problems::Array{T, N}
end

function Scenario(
    ;
    n::Int = 2,
    varying::Symbol,
    varying2::Union{Symbol, Nothing} = nothing,
    kwargs...
)
    @assert haskey(kwargs, varying) "Varying param $varying must be in kwargs"
    kwargs_ = Dict(
        k => if k == varying || k == varying2
                if v isa AbstractMatrix
                    @assert size(v, 2) == n "$k must have n columns"
                    [x for x in eachrow(v)]
                elseif v isa AbstractVector
                    collect(v)
                else
                    [v]
                end
             else
                v
             end
        for (k, v) in kwargs
    )

    n_steps = length(kwargs_[varying])
    n_steps2 = if isnothing(varying2)
        1
    else
        length(kwargs_[varying2])
    end

    problems = if isnothing(varying2)
        get_problems(n, kwargs_, varying)
    else
        get_problems_with_secondary_variation(n, kwargs_, varying, varying2)
    end

    return Scenario(
        n, n_steps, n_steps2,
        varying, varying2,
        kwargs_[varying], isnothing(varying2) ? nothing : kwargs_[varying2],
        problems
    )
end

function get_problems(n, kwargs_in, varying)
    n_steps = length(kwargs_in[varying])
    kwargs_list = [
        Dict(
            k => if k == varying
                    v[i]
                else
                    v
                end
            for (k, v) in kwargs_in
        )
        for i in 1:n_steps
    ]
    return [Problem(n = n; kwargs...) for kwargs in kwargs_list]
end

function get_problems_with_secondary_variation(n, kwargs_in, varying, varying2)
    n_steps = length(kwargs_in[varying])
    n_steps_secondary = length(kwargs_in[varying2])
    problems = Array{Problem, 2}(undef, n_steps, n_steps_secondary)
    for i in 1:n_steps, j in 1:n_steps_secondary
        kwargs = Dict(
            k => if k == varying
                    v[i]
                elseif k == varying2
                    v[j]
                else
                    v
                end
            for (k, v) in kwargs_in
        )
        problems[i, j] = Problem(n = n; kwargs...)
    end
    return problems
end

function get_problem(scenario::Scenario, index)
    return scenario.problems[index]
end

function extract(scenario::Scenario, field::Symbol)
    out = Array{Any}(undef, scenario.n_steps, scenario.n_steps2)
    for i in 1:scenario.n_steps, j in 1:scenario.n_steps2
        problem = scenario.problems[i, j]
        out[i, j] = if field in (:A, :α, :B, :β, :θ)
            getfield(problem.prodFunc, field)
        else
            getfield(problem, field)
        end
    end
    return out
end



struct ScenarioResult{T <: Union{SolverResult, Vector{SolverResult}}, N}
    scenario::Scenario
    solverResults::Array{T, N}
end


function extract(res::ScenarioResult{SolverResult, 1}, field::Symbol)
    return if field in (:success, :Xs, :Xp, :s, :p, :σ, :payoffs)
        [getfield(x, field) for x in res.solverResults]
    else
        getfield(res.scenario, field)
    end
end

function extract(res::ScenarioResult{Vector{SolverResult}, 1}, field::Symbol)
    return if field in (:success, :Xs, :Xp, :s, :p, :σ, :payoffs)
        [[getfield(x, field) for x in s] for s in res.solverResults]
    else
        getfield(res.scenario, field)
    end
end


# SOLVER FUNCTIONS:

function setup_results(scenario::Scenario, method)
    if !isnothing(scenario.varying2)
        @assert method != :scatter && method != :mixed "Secondary variation is unsupported with the mixed or scatter solvers."

        return if method in (:mixed, :scatter)
            Array{Vector{SolverResult}}(undef, (scenario.n_steps, scenario.n_steps2))
        else
            Array{SolverResult}(undef, (scenario.n_steps, scenario.n_steps2))
        end
    else
        return if method in (:mixed, :scatter)
            Array{Vector{SolverResult}}(undef, scenario.n_steps)
        else
            Array{SolverResult}(undef, scenario.n_steps)
        end
    end
end

function fill_results!(results, scenario, method, options, indices = nothing)
    if isnothing(indices)
        indices = CartesianIndices(size(results))
    end
    Threads.@threads for index in indices
        problem = get_problem(scenario, index)
        results[index] = solve(problem, method, options)
    end
end

function is_jump(x, pred, succ)
    if !x.success
        return true
    end
    # if x is about the same as its predecessor, any problem is probably in successor
    if !isapprox(x.Xs, pred.Xs, rtol = 0.25) && !isapprox(x.Xp, pred.Xp, rtol = 0.25)
        return false
    end
    # figure out where x would be if it fell directly between predecessor and successor
    # if it's too far from that, it's a jump
    est_Xs = (pred.Xs .+ succ.Xs) ./ 2
    est_Xp = (pred.Xp .+ succ.Xp) ./ 2
    # value needs to be double its predicted value (or vice versa) to be considered a jump
    if !isapprox(est_Xs, x.Xs, rtol = 0.5) || !isapprox(est_Xp, x.Xp, rtol = 0.5)
        return true
    end
    return false
end

function find_jumps(results::Vector)
    idxs = Vector{CartesianIndex}()
    for i in axes(results, 1)[2:end-1]
        if is_jump(results[i], results[i-1], results[i+1])
            println("Found jump at $i")
            push!(idxs, CartesianIndex(i))
        end
    end
    return idxs
end

function find_jumps(results)
    idxs = Vector{CartesianIndex}()
    (m, n) = size(results)
    # each column corresponds to a different value of the secondary varying param
    # results should vary continuously within each column
    for j in 1:n, i in 2:(m-1)
        if is_jump(results[i, j], results[i-1, j], results[i+1, j])
            println("Found jump at $((i, j))")
            push!(idxs, CartesianIndex(i, j))
        end
    end
    return idxs
end

function enforce_continuity!(results, scenario, method, options, retries = 10)
    # find results that jump a lot relative to their neighbors
    # re-do those until they're close to their neighbors
    idxs_to_redo = find_jumps(results)
    if !isempty(idxs_to_redo)
        fill_results!(results, scenario, method, options, idxs_to_redo)
        if retries > 0
            enforce_continuity!(results, scenario, method, options, retries - 1)
        else
            println("Hit max jump retries")
        end
    end
end


function solve(scenario::AbstractScenario, method, options)
    
    # create an empty array of the correct size to contain the results
    results = setup_results(scenario, method)
    # send to solver
    fill_results!(results, scenario, method, options)
    # check for jumps (likely errors) and re-solve at those points
    # Doesn't quite work yet, so commenting out for now
    # enforce_continuity!(results, scenario, method, options)

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
    beliefs::Vector{Dict{Symbol, Any}}
end

function ScenarioWithBeliefs(
    baseScenario = Scenario();
    beliefs = fill(Dict(), Scenario().n)
)
    n = baseScenario.n
    @assert length(beliefs) == n
    # expand beliefs so values are vectors of length n
    beliefs_ = [
        Dict{Symbol, Any}(
            k => as_Float64_Array(v, n) for (k, v) in b
        )
        for b in beliefs
    ]
    @assert all(length(x) == n for b in beliefs_ for x in values(b))
    return ScenarioWithBeliefs(baseScenario, beliefs_)
end

function setup_results(scenario::ScenarioWithBeliefs, method)
    return setup_results(scenario.baseScenario, method)
end

function get_problem(scenario::ScenarioWithBeliefs, index)
    idx = Tuple(index)
    if length(idx) == 1
        @assert isnothing(scenario.baseScenario.varying2)
    end

    baseProblem = scenario.baseScenario.problems[index]

    return ProblemWithBeliefs(baseProblem, scenario.beliefs)
end

function ScenarioResult(scenario::ScenarioWithBeliefs, results)
    return ScenarioResult(scenario.baseScenario, results)
end
