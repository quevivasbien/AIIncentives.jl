struct Problem{T <: Real}
    n::Integer
    d::Vector{T}
    r::Vector{T}
    prodFunc::ProdFunc{T}
    riskFunc::RiskFunc
    csf::CSF
end

function Problem(
    ;
    n::Integer = 2,
    d::Union{Real, AbstractVector} = 0.,
    r::Union{Real, AbstractVector} = 0.1,
    RiskFunc::Union{RiskFunc, Nothing} = nothing,
    prodFunc::ProdFunc{Float64} = ProdFunc(),
    csf::CSF = CSF()
)
    @assert n >= 2 "n must be at least 2"
    @assert n == prodFunc.n "n must match prodFunc.n"
    problem = Problem(
        n,
        as_Float64_Array(d, n),
        as_Float64_Array(r, n),
        prodFunc,
        isnothing(riskFunc) ? MultiplicativeRiskFunc(n) : riskFunc,
        csf
    )
    @assert all(length(getfield(problem, x)) == n for x in [:d, :r]) "Your input params need to match the number of players"
    return problem
end

function payoff(problem::Problem, i::Integer, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    σ = get_total_safety(problem.riskFunc, s, p)
    return σ .* reward(problem.csf, i, p) .- (1. .- σ) .* problem.d[i] .- problem.r[i] .* (Xs[i] + Xp[i])
end

function all_payoffs_with_s_p(problem::Problem, Xs::Vector, Xp::Vector, s::Vector, p::Vector)
    σ = get_total_safety(problem.riskFunc, s, p)
    return σ .* all_rewards(problem.csf, p) .- (1. .- σ) .* problem.d .- problem.r .* (Xs .+ Xp)
end

function all_payoffs(problem::Problem, Xs::Vector, Xp::Vector)
    (s, p) = f(problem.prodFunc, Xs, Xp)
    return all_payoffs_with_s_p(problem, Xs, Xp, s, p)
end

function get_func(problem::Problem, i::Integer, strats::Array)
    strats_ = copy(strats)
    function func(x)
        strats_[i, :] = x
        return -payoff(problem, i, strats_[:, 1], strats_[:, 2])
    end
end

function is_symmetric(problem::Problem)
    (
        is_symmetric(problem.prodFunc)
        && all(problem.d[1] .== problem.d[2:problem.n])
        && all(problem.r[1] .== problem.r[2:problem.n])
    )
end
