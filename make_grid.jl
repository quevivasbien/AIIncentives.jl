using DataFrames
include("scenarios.jl")

function make_grid(
    ;
    n_players = 2,
    A_range = 10 .^ range(1, 3, length = 3),
    α_range = range(0.1, 0.9, step = 0.2),
    B_range = 10 .^ range(1, 3, length = 3),
    β_range = range(0.1, 0.9, step = 0.2),
    θ_range = range(0, 2, step = 0.25),
    d_range = [0., 1.],
    r_range = 10 .^ range(-2, 1, length = 8),
    options = SolverOptions()
)
    (A, α, B, β, θ, d, r) = Tuple(
        getindex.(
            reshape(
                Iterators.product(
                    A_range, α_range, B_range, β_range, θ_range, d_range, r_range
                ) |> collect,
                :, 1
            ), i
        ) |> vec for i in 1:7
    )
    n_values = length(A)
    success = Vector{Bool}(undef, n_values)
    Xs = Array{Float64}(undef, n_values, 2)
    Xp = Array{Float64}(undef, n_values, 2)
    s = Array{Float64}(undef, n_values, 2)
    p = Array{Float64}(undef, n_values, 2)
    σ = Vector{Float64}(undef, n_values)
    payoffs = Array{Float64}(undef, n_values, 2)
    println("Iterating over $n_values value combinations...")
    count = 0
    Threads.@threads for i in 1:n_values
        prodFunc = ProdFunc(
            n_players,
            fill(A[i], n_players),
            fill(α[i], n_players),
            fill(B[i], n_players),
            fill(β[i], n_players),
            fill(θ[i], n_players)
        )
        problem = Problem(n_players, fill(d[i], n_players), fill(r[i], n_players), prodFunc, CSF())
        result = solve_iters(problem, options)
        success[i] = result.success
        Xs[i, :] = result.Xs
        Xp[i, :] = result.Xp
        s[i, :] = result.s
        p[i, :] = result.p
        σ[i] = get_total_safety(result.s)
        payoffs[i, :] = result.payoffs

        count += 1
        if count % 1000 == 0 || count == n_values
            println("Completed $count of $n_values")
        end
    end

    DataFrame(
        A = A,
        alpha = α,
        B = B,
        beta = β,
        theta = θ,
        d = d,
        r = r,
        success = success,
        Xs1 = Xs[:, 1],
        Xs2 = Xs[:, 2],
        Xp1 = Xp[:, 1],
        Xp2 = Xp[:, 2],
        s1 = s[:, 1],
        s2 = s[:, 2],
        p1 = p[:, 1],
        p2 = p[:, 2],
        proba_safe = σ,
        payoff1 = payoffs[:, 1],
        payoff2 = payoffs[:, 2]
    )
end

function get_problem(row::DataFrameRow)
    prodFunc = ProdFunc(
        A = fill(row[:A], 2),
        α = fill(row[:alpha], 2),
        B = fill(row[:B], 2),
        β = fill(row[:beta], 2),
        θ = fill(row[:theta], 2)
    )
    Problem(
        d = fill(row[:d], 2),
        r = fill(row[:r], 2),
        prodFunc = prodFunc
    )
end

function visual_check(row::DataFrameRow)
    problem = get_problem(row)

    base_Xs = [row[:Xs1], row[:Xs2]]
    base_Xp = [row[:Xp1], row[:Xp2]]

    plot(
        plot_payoffs_with_xs(
            problem, base_Xs, base_Xp, 1,
            min_Xp = base_Xp[1] / 10, max_Xp = base_Xp[1] * 2 + EPSILON
        ),
        plot_payoffs_with_xp(
            problem, base_Xs, base_Xp, 1,
            min_Xs = base_Xs[1] / 10, max_Xs = base_Xs[1] * 2 + EPSILON
        ),
        plot_payoffs_with_xs(
            problem, base_Xs, base_Xp, 2,
            min_Xp = base_Xp[2] / 10, max_Xp = base_Xp[2] * 2 + EPSILON
        ),
        plot_payoffs_with_xp(
            problem, base_Xs, base_Xp, 2,
            min_Xs = base_Xs[2] / 10, max_Xs = base_Xs[2] * 2 + EPSILON
        ),
        layout = (2, 2)
    )
end