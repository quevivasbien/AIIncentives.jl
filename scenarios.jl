using Plots
include("./model.jl")


struct Scenario
    n_players::Integer
    A::Array
    α::Array
    B::Array
    β::Array
    θ::Array
    d::Array
    r::Array
    w::Number
    l::Number
    a_w::Number
    a_l::Number
    varying_param
    secondary_varying_param
end

Scenario(
    n_players,
    A, α, B, β, θ,
    d, r;
    w = 1., l = 0., a_w = 0., a_l = 0.,
    varying_param = :r,
    secondary_varying_param = nothing
) = Scenario(n_players, A, α, B, β, θ, d, r, w, l, a_w, a_l, varying_param, secondary_varying_param)


function solve_with_secondary_variation(
    scenario::Scenario,
    plot,
    plotname,
    labels,
    title,
    logscale,
    solve_method,
    max_iters,
    tol
)
    # To-Do
    println("Not yet implemented!")
    return(get_null_result())
end


function solve(
    scenario::Scenario;
    plot = true,
    plotname = "scenario",
    labels = nothing,
    title = nothing,
    logscale = false,
    solve_method = "hybrid",
    max_iters::Integer = 100,
    tol::Number = 1e-8,
)
    if scenario.secondary_varying_param != nothing
        return solve_with_secondary_variation(scenario, plot, plotname, labels, title, logscale, solve_method, max_iters, tol)
    end

    if labels != nothing && length(labels) != scenario.n_players
        println("Length of labels should match number of players")
        return get_null_result()
    end

    varying_param = getfield(scenario, scenario.varying_param)
    n_steps = size(varying_param)[2]
    
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
        var = getfield(scenario, symbol)
        if symbol == scenario.varying_param
            copy!(newvar, var)
        else
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

    results = SolverResult[]
    csf = CSF(scenario.w, scenario.l, scenario.a_w, scenario.a_l)
    # send to solver
    for i in 1:n_steps
        prodFunc = ProdFunc(A[:, i], α[:, i], B[:, i], β[:, i], θ[:, i])
        problem = Problem(d[:, i], r[:, i],  prodFunc, csf)
        if solve_method == "iters"
            push!(
                results,
                solve_iters(problem, max_iters = max_iters, tol = tol)
            )
        elseif solve_method == "roots"
            push!(
                results,
                solve_roots(problem)
            )
        else
            # implicitly solve_method == "hybrid" by default
            push!(
                results,
                solve_hybrid(
                    problem,
                    max_iters = max_iters,
                    tol = tol
                )
            )
        end
    end

    return results
end


function test()
    A = [10., 10.]
    α = [0.5, 0.5]
    B = [10., 10.]
    β = [0.5, 0.5]
    θ = [0., 0.]
    d = [1., 1.]
    r = repeat([0.1 0.2 0.3], 2)

    scenario = Scenario(2, A, α, B, β, θ, d, r)

    @time result = solve(scenario)
    println(result)
end
