struct SolverResult{T <: Real}
    success::Bool
    Xs::Vector{T}
    Xp::Vector{T}
    s::Vector{T}
    p::Vector{T}
    σ::T
    payoffs::Vector{T}
end

function prune_duplicates(results::Vector{SolverResult}; atol = 1e-6, rtol=5e-2)
    dups = Vector{Int}()
    unique = Vector{Int}()
    n_results = length(results)
    for i in 1:n_results
        if i in dups
            continue
        end
        strats1 =  hcat(results[i].Xs, results[i].Xp)
        for j in (i+1):n_results
            strats2 = hcat(results[j].Xs, results[j].Xp)
            if isapprox(strats1, strats2; atol = atol, rtol = rtol)
                push!(dups, j)
            end
        end
        push!(unique, i)
    end
    return results[unique]
end

function get_s_p_σ_payoffs(problem::Problem, Xs_, Xp_)
    (s, p) = f(problem.prodFunc, Xs_, Xp_)
    σ = get_total_safety(problem.riskFunc, s, p)
    payoffs = all_payoffs_with_s_p(problem, Xs_, Xp_, s, p)
    return s, p, σ, payoffs
end

function SolverResult(problem::Problem, success::Bool, Xs::Vector{T}, Xp::Vector{T}; fill = true) where {T <: Real}
    return if fill
        SolverResult(success, Xs, Xp, get_s_p_σ_payoffs(problem, Xs, Xp)...)
    else
        SolverResult(success, Xs, Xp, similar(Xs), similar(Xs), undef, similar(Xs))
    end
end

function get_null_result(n)
    return SolverResult(
        false,
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n)
    )
end

function resolve_multiple_solutions(
    results::Vector{SolverResult},
    problem::Problem
)
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

function print(result::SolverResult)
    println("success: ", result.success)
    println("Xs: ", result.Xs)
    println("Xp: ", result.Xp)
    println("s: ", result.s)
    println("p: ", result.p)
    println("σ: ", result.σ)
    println("payoffs: ", result.payoffs, '\n')
end