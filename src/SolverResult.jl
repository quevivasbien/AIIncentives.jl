struct SolverResult{T <: AbstractFloat, N}
    success::Bool
    Xs::Array{T, N}
    Xp::Array{T, N}
    s::Array{T, N}
    p::Array{T, N}
    payoffs::Array{T, N}
end

function trim_to_index(result::SolverResult, index)
    return SolverResult(
        result.success,
        copy(selectdim(result.Xs, 1, index)),
        copy(selectdim(result.Xp, 1, index)),
        copy(selectdim(result.s, 1, index)),
        copy(selectdim(result.p, 1, index)),
        copy(selectdim(result.payoffs, 1, index))
    )
end

function prune_duplicates(result::SolverResult{T, 1}; atol = 1e-6, rtol=5e-2) where T
    return result
end

function prune_duplicates(result::SolverResult; atol = 1e-6, rtol=5e-2)
    dups = Vector{Integer}()
    unique = Vector{Integer}()
    n_results = size(result.Xs, 1)
    for i in 1:n_results
        if i ∈ dups
            continue
        end
        strats1 =  hcat(selectdim(result.Xs, 1, i), selectdim(result.Xp, 1, i))
        for j in (i+1):n_results
            strats2 = hcat(selectdim(result.Xs, 1, j), selectdim(result.Xp, 1, j))
            if isapprox(strats1, strats2; atol = atol, rtol = rtol)
                push!(dups, j)
            end
        end
        push!(unique, i)
    end
    return trim_to_index(result, unique)
end

function get_s_p_payoffs(problem::Problem, Xs_, Xp_)
    if ndims(Xs_) > 1
        Xs = reshape(Xs_, :, problem.n)
        Xp = reshape(Xp_, :, problem.n)
        s = similar(Xs)
        p = similar(Xs)
        payoffs = similar(Xs)
        for i in 1:size(Xs)[1]
            (s[i, :], p[i, :]) = f(problem.prodFunc, Xs[i, :], Xp[i, :])
            payoffs[i, :] = all_payoffs_with_s_p(problem, Xs[i, :], Xp[i, :], s[i, :], p[i, :])
        end
        return (
            reshape(s, size(Xs_)),
            reshape(p, size(Xs_)),
            reshape(payoffs, size(Xs_))
        )
    else
        (s, p) = f(problem.prodFunc, Xs_, Xp_)
        payoffs = all_payoffs_with_s_p(problem, Xs_, Xp_, s, p)
        return s, p, payoffs
    end
end

function SolverResult(problem::Problem, success::Bool, Xs, Xp; fill = true, prune = true)
    result = if fill
        SolverResult(success, Xs, Xp, get_s_p_payoffs(problem, Xs, Xp)...)
    else
        SolverResult(success, Xs, Xp, similar(Xs), similar(Xs), similar(Xs))
    end
    if prune
        return prune_duplicates(result)
    else
        return result
    end
end

function make_2d(result::SolverResult{T, 2}) where T
    return result
end

function make_2d(result::SolverResult, n)
    return SolverResult(
        result.success,
        reshape(result.Xs, :, n),
        reshape(result.Xp, :, n),
        reshape(result.s, :, n),
        reshape(result.p, :, n),
        reshape(result.payoffs, :, n)
    )
end

function Base.:+(result1::SolverResult, result2::SolverResult)
    return SolverResult(
        result1.success && result2.success,
        cat(result1.Xs, result2.Xs, dims = 1),
        cat(result1.Xp, result2.Xp, dims = 1),
        cat(result1.s, result2.s, dims = 1),
        cat(result1.p, result2.p, dims = 1),
        cat(result1.payoffs, result2.payoffs, dims = 1)
    )
end

function get_null_result(n)
    return SolverResult(
        false,
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n)
    )
end

function resolve_multiple_solutions(
    result::SolverResult,
    problem::Problem
)
    if ndims(result.Xs) == 1
        return result
    elseif size(result.Xs)[1] == 1
        return trim_to_index(result, 1)
    end

    argmaxes = [x[1] for x in argmax(result.payoffs, dims=1)]
    best = argmaxes[1]
    if any(best .!= argmaxes[2:end])
        println("More than one result found; equilibrium is ambiguous")
        return get_null_result(problem.n)
    end
    
    return trim_to_index(result, best)
end

function print(result::SolverResult)
    println("success: ", result.success)
    println("Xs: ", result.Xs)
    println("Xp: ", result.Xp)
    println("s: ", result.s)
    println("p: ", result.p)
    println("payoffs: ", result.payoffs, '\n')
end