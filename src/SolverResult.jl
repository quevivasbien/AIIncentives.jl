struct SolverResult{N}
    success::Bool
    Xs::SVector{N, Float64}
    Xp::SVector{N, Float64}
    s::SVector{N, Float64}
    p::SVector{N, Float64}
    σ::Float64
    payoffs::SVector{N, Float64}
end

function SolverResult(
    success::Bool,
    Xs::AbstractVector,
    Xp::AbstractVector,
    s::AbstractVector,
    p::AbstractVector,
    σ::Real,
    payoffs::AbstractVector,
)
    n = length(Xs)
    return SolverResult(
        success,
        SVector{n, Float64}(Xs),
        SVector{n, Float64}(Xp),
        SVector{n, Float64}(s),
        SVector{n, Float64}(p),
        Float64(σ),
        SVector{n, Float64}(payoffs)
    )
end

function SolverResult(problem::AbstractProblem{N}, success::Bool, Xs::AbstractVector, Xp::AbstractVector, fill = true) where {N}
    Xs = SVector{N, Float64}(Xs)
    Xp = SVector{N, Float64}(Xp)
    return if fill
        SolverResult{N}(success, Xs, Xp, s_p_σ_payoffs(problem, Xs, Xp)...)
    else
        SolverResult{N}(success, Xs, Xp, similar(Xs), similar(Xs), undef, similar(Xs))
    end
end

function get_null_result(n)
    return SolverResult(
        false,
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        NaN,
        fill(NaN, n)
    )
end

function resolve_multiple_solutions(
    results::Vector{SolverResult{N}},
    problem::AbstractProblem{N}
) where {N}
    if length(results) == 1
        return results[1]
    end

    argmaxes = [x[1] for x in argmax(result.payoffs, dims=1)]
    best = argmaxes[1]
    if any(best .!= argmaxes[2:end])
        println("More than one result found; equilibrium is ambiguous")
        return get_null_result(problem.n)
    end
    
    return results[best]
end

function resolve_multiple_solutions(results, ::ProblemWithBeliefs{N}) where {N}
    # can't distinguish between results with different beliefs
    if length(results) > 1
        println("More than one result found; equilibrium is ambiguous")
        return get_null_result(N)
    end
    return results[1]
end

function print(result::SolverResult)
    println("success: ", result.success)
    println("Xs: ", result.Xs)
    println("Xp: ", result.Xp)
    println("s: ", result.s)
    println("p: ", result.p)
    println("σ: ", result.σ)
    println("payoffs: ", result.payoffs, '\n')
end


"""
For results of solve_mixed, draws results from equilibrium distribution

X has shape (n, 2, t) where t is number of samples used in solve (= options.n_points, not the same as n_samples)
"""
function sample_from(problem::AbstractProblem, X::AbstractArray, n_samples = 100)
    (n, _, t) = size(X)
    samples = [
        begin
            indices = rand(1:t, n)
            [X[i, 1, indices[i]] for i in 1:n], [X[i, 2, indices[i]] for i in 1:n]
        end
        for _ in 1:n_samples
    ]

    [
        SolverResult(problem, true, xs, xp)
        for (xs, xp) in samples
    ]
end
