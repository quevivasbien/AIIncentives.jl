"""
Abstract type to allow insertion of alternate problem types

Subtypes should implement `get_payoff`, `payoffs_with_s_p`, `get_payoffs`, `get_s_p_σ_payoffs`, `is_symmetric` and have an integer field `n`
"""
abstract type AbstractProblem end


(problem::AbstractProblem)(i::Int, Xs::Vector, Xp::Vector) = get_payoff(problem, i, Xs, Xp)

(problem::AbstractProblem)(Xs::Vector, Xp::Vector) = get_payoffs(problem, Xs, Xp)

function get_func(problem::AbstractProblem, i::Int, strats::Array)
    strats_ = copy(strats)
    function func(x)
        strats_[i, :] = x
        return -get_payoff(problem, i, strats_[:, 1], strats_[:, 2])
    end
end

@doc raw"""
The `Problem.jl` file implements a `Problem` type that represents the payoff function

$$u_i := \sum_{j=1}^n \sigma_j(s) q_j(p) \rho_{ij}(p) - \left( 1 - \sum_{j=1}^n \sigma_j(s) q_j(p) \right) d_i - r_i(X_{i,s} + X_{i,p})$$

You can construct a `Problem` like this:
```julia
problem = Problem(
    n_players = 2,  # default is 2
    d = 1.,
    r = 0.05,
    prodFunc = yourProdFunc,
    riskFunc = yourRiskFunc,  # default is WinnerOnlyRisk()
    csf = yourCSF,  # default is BasicCSF()
    payoffFunc = yourPayoffFunc  # default is LinearPayoff(1, 0, 0, 0)
)
```
Note that the lengths of `d` and `r` must match and be equal to `n` and `prodFunc.n_players`. Again, you can omit arguments to use default values or provide vectors instead of scalars if you want different values for each player.

To calculate the payoffs for all the players, you can do
```julia
payoffs = get_payoffs(problem, Xs, Xp)
```
or
```julia
payoffs = problem(Xs, Xp)
```

and for just player `i`,
```julia
payoff_i = get_payoff(problem, i, Xs, Xp)
```
or
```julia
payoff_i = problem(i, Xs, Xp)
```

(Note that in the above, `Xs` and `Xp` are vectors of length `problem.n`.)
"""
struct Problem{T <: Real, R <: RiskFunc, C <: CSF, P <: PayoffFunc, K <: CostFunc} <: AbstractProblem
    n::Int
    d::Vector{T}
    prodFunc::ProdFunc{T}
    riskFunc::R
    csf::C
    payoffFunc::P
    costFunc::K
end

function Problem(
    n::Int,
    d::Vector{T},
    r::Vector{T},
    prodFunc::ProdFunc{T},
    riskFunc::R,
    csf::C,
    payoffFunc::P,
) where {T <: Real, R <: RiskFunc, C <: CSF, P <: PayoffFunc}
    return Problem(n, d, prodFunc, riskFunc, csf, payoffFunc, FixedUnitCost(n, r))
end

function Problem(
    ;
    n::Int = 2,
    d::Union{Real, AbstractVector} = 0.,
    r::Union{Real, AbstractVector} = 0.1,
    riskFunc::RiskFunc = WinnerOnlyRisk(),
    prodFunc::ProdFunc{Float64} = ProdFunc(),
    csf::CSF = BasicCSF(),
    payoffFunc::PayoffFunc = LinearPayoff(),
    costFunc::Union{Nothing, CostFunc} = nothing,
)
    @assert n >= 2 "n must be at least 2"
    @assert n == prodFunc.n "n must match prodFunc.n"
    @assert n == payoffFunc.n "n must match payoffFunc.n"
    if isnothing(costFunc)
        costFunc = FixedUnitCost(n, as_Float64_Array(r, n))
    end
    @assert n == costFunc.n "n must match costFunc.n"
    problem = Problem(
        n,
        as_Float64_Array(d, n),
        prodFunc,
        riskFunc,
        csf,
        payoffFunc,
        costFunc
    )
    return problem
end

function get_payoff(problem::Problem, i::Int, Xs::Vector, Xp::Vector)
    (s, p) = problem.prodFunc(Xs, Xp)
    proba_win = problem.csf(p)  # probability that each player wins
    (pf_win, pf_lose) = problem.payoffFunc(i, p[i])  # payoffs if player i wins/loses
    payoffs = [(j == i) ? pf_win : pf_lose for j in 1:problem.n]
    σis = problem.riskFunc(s)  # vector of proba(safe) conditional on each player winning
    cond_σ = proba_win .* σis
    if problem.payoffFunc isa PayoffOnDisaster && problem.payoffFunc.whogets[i]
        return sum(payoffs .* cond_σ) + sum((payoffs .- problem.d[i]) .* (1 .- cond_σ)) - problem.costFunc(i, Xs, Xp)
    else
        return sum(payoffs .* cond_σ) - (1 - sum(cond_σ)) * problem.d[i] - problem.costFunc(i, Xs, Xp)
    end
end

function payoffs_with_s_p(problem::Problem, Xs::Vector, Xp::Vector, s::Vector, p::Vector)
    proba_win = problem.csf(p)  # probability that each player wins
    # construct matrix of payoffs
    (pf_win, pf_lose) = problem.payoffFunc(p)  # payoffs if each player wins/loses
    payoffs = repeat(pf_lose, 1, problem.n)
    payoffs[diagind(payoffs)] = pf_win
    σis = problem.riskFunc(s)  # vector of proba(safe) conditional on each player winning
    cond_σ = proba_win .* σis  # proba that i wins and outcome is safe
    if problem.payoffFunc isa PayoffOnDisaster
        safe_payoffs = vec(sum(payoffs .* cond_σ, dims = 2))
        cond_d = proba_win .* (1 .- σis)  # proba that i wins and outcome is not safe
        disaster_payoffs = vec(
            sum((payoffs .* problem.payoffFunc.whogets .- problem.d) .* cond_d,
            dims = 2
        ))
        return safe_payoffs .+ disaster_payoffs .- problem.costFunc(Xs, Xp)
    else
        return vec(sum(payoffs .* cond_σ, dims = 2)) .- (1 .- sum(cond_σ)) .* problem.d .- problem.costFunc(Xs, Xp)
    end
end

function get_payoffs(problem::Problem, Xs::Vector, Xp::Vector)
    (s, p) = problem.prodFunc(Xs, Xp)
    return payoffs_with_s_p(problem, Xs, Xp, s, p)
end

function get_s_p_σ_payoffs(problem::Problem, Xs_, Xp_)
    (s, p) = f(problem.prodFunc, Xs_, Xp_)
    # proba(safe) = sum(proba(i safe) * proba(i win))
    σ = sum(problem.riskFunc(s) .* problem.csf(p))
    payoffs = payoffs_with_s_p(problem, Xs_, Xp_, s, p)
    return s, p, σ, payoffs
end

function is_symmetric(problem::Problem)
    (
        is_symmetric(problem.prodFunc)
        && is_symmetric(problem.costFunc)
        && all(problem.d[1] .== problem.d[2:problem.n])
    )
end


"""
Meant to model cases where players make decisions based on possibly incorrect beliefs about model params

It is assumed that players know each other's beliefs (though maybe nobody has correct beliefs), otherwise we end up in the crazy land of higher-order beliefs

Contains a true Problem `baseProblem`, but also a vector `beliefs` of Problems, which are the Problems that each player thinks they are dealing with
"""
struct ProblemWithBeliefs{T <: Problem} <: AbstractProblem
    n::Int
    baseProblem::T
    beliefs::Vector{T}
end

function ProblemWithBeliefs(
    baseProblem::T = Problem();
    beliefs::Vector{T} = fill(Problem(), Problem().n)
) where {T <: Problem}
    n = baseProblem.n
    @assert n == length(beliefs) "Length of beliefs needs to match baseProblem.n"
    @assert all(n == problem.n for problem in beliefs) "Beliefs need to have same n as base problem"
    return ProblemWithBeliefs(n, baseProblem, beliefs)
end


# pass calls to baseProblem automatically
macro to_baseProblem(f)
    fname = Symbol(f)
    quote
        function $(esc(fname))(problem::ProblemWithBeliefs, args...)
            return $(esc(fname))(problem.baseProblem, args...)
        end     
    end
end

@to_baseProblem get_payoff
@to_baseProblem get_payoffs
@to_baseProblem payoffs_with_s_p
@to_baseProblem get_s_p_σ_payoffs

is_symmetric(problem::ProblemWithBeliefs) = (
    is_symmetric(problem.baseProblem)
    && is_symmetric(problem.beliefs[1])
    && all(problem.beliefs[1] == b for b in problem.beliefs[2:end])
)
