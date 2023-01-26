"""
Abstract type to allow insertion of alternate problem types

Subtypes should implement `payoff`, `payoffs_with_s_p`, `payoffs`, `s_p_σ_payoffs`, `is_symmetric` and have an integer field `n`
"""
abstract type AbstractProblem{N} end


(problem::AbstractProblem)(i::Int, Xs::AbstractVector, Xp::AbstractVector) = payoff(problem, i, Xs, Xp)

(problem::AbstractProblem)(Xs::AbstractVector, Xp::AbstractVector) = payoffs(problem, Xs, Xp)

function get_func(problem::AbstractProblem{N}, i::Int, strats::AbstractArray) where N
    strats_ = MMatrix{N, 2}(deepcopy(strats))
    function func(x)
        strats_[i, :] = x
        return -payoff(problem, i, strats_[:, 1], strats_[:, 2])
    end
end

@doc raw"""
The `Problem.jl` file implements a `Problem` type that represents the payoff function

$$u_i := \sum_{j=1}^n \sigma_j(s) q_j(p) \rho_{ij}(p) - \left( 1 - \sum_{j=1}^n \sigma_j(s) q_j(p) \right) d_i - c_i(X_s, X_p)$$

You can construct a `Problem` like this:
```julia
problem = Problem(
    n = 2,  # default is 2
    d = 1.,
    r = 0.05,
    prodFunc = yourProdFunc,
    riskFunc = yourRiskFunc,  # default is WinnerOnlyRisk()
    csf = yourCSF,  # default is BasicCSF()
    payoffFunc = yourPayoffFunc,  # default is LinearPayoff(1, 0, 0, 0)
)
```
Note that the lengths of `d` and `r` must match and be equal to `n` and `prodFunc.n`. Again, you can omit arguments to use default values or provide vectors instead of scalars if you want different values for each player.

Instead of providing `r`, you can provide a `CostFunc` with the keyword `costFunc`, e.g., `costFunc = FixedUnitCost(2, [0.1, 0.1])`. If you just provide `r`, it will be interpreted as `costFunc = FixedUnitCost(n, r)`.

To calculate the payoffs for all the players, you can do
```julia
payoffs(problem, Xs, Xp)
```
or
```julia
problem(Xs, Xp)
```

and for just player `i`,
```julia
payoff(problem, i, Xs, Xp)
```
or
```julia
problem(i, Xs, Xp)
```

(Note that in the above, `Xs` and `Xp` are vectors of length N.)
"""
struct Problem{N, R <: RiskFunc, C <: CSF, P <: PayoffFunc{N}, K <: CostFunc{N}} <: AbstractProblem{N}
    d::SVector{N, Float64}
    prodFunc::ProdFunc{N}
    riskFunc::R
    csf::C
    payoffFunc::P
    costFunc::K
end

"""
constructor for problem, where defaults will be overridden by provided kwargs
    mostly just a helper, probably shouldn't be called by user
"""
function Problem(
    default_d,
    default_costFunc,
    default_prodFunc,
    default_riskFunc,
    default_csf,
    default_payoffFunc
    ;
    n::Int = 2,
    kwargs...
)
    @assert n >= 2
    
    d = if haskey(kwargs, :d)
        as_SVector(kwargs[:d], n)
    else
        default_d
    end

    prodFunc = if haskey(kwargs, :prodFunc)
        kwargs[:prodFunc]
    else
        prodFunc_kwargs = (
            key => val for (key, val) in kwargs
                if key in (:A, :α, :alpha, :B, :β, :beta, :θ, :theta)
        )
        if isempty(prodFunc_kwargs)
            default_prodFunc
        else
            ProdFunc(default_prodFunc; prodFunc_kwargs...)
        end
    end
    @assert prodFunc isa ProdFunc{n}

    costFunc = if haskey(kwargs, :costFunc)
        kwargs[:costFunc]
    elseif haskey(kwargs, :rs) && haskey(kwargs, :rp)
        FixedUnitCost2(as_MVector(kwargs[:rs], n), as_MVector(kwargs[:rp], n))
    elseif haskey(kwargs, :r0) && haskey(kwargs, :r1) && haskey(kwargs, :s_thresh)
        CertificationCost(as_MVector(kwargs[:r0], n), as_MVector(kwargs[:r1], n), as_MVector(kwargs[:s_thresh], n), prodFunc)
    elseif haskey(kwargs, :r0) && haskey(kwargs, :r1) && haskey(kwargs, :δ)
        RelativeSafetyCertificationCost(as_MVector(kwargs[:r0], n), as_MVector(kwargs[:r1], n), as_MVector(kwargs[:δ], n), prodFunc)
    elseif haskey(kwargs, :r) && haskey(kwargs, :δ) && haskey(kwargs, :s_thresh)
        PenaltyCost(as_MVector(kwargs[:r], n), as_MVector(kwargs[:δ], n), as_MVector(kwargs[:s_thresh], n), prodFunc)
    elseif haskey(kwargs, :r)
        FixedUnitCost(as_MVector(kwargs[:r], n))
    else
        default_costFunc
    end
    @assert costFunc isa CostFunc{n}

    riskFunc = if haskey(kwargs, :riskFunc)
        kwargs[:riskFunc]
    else
        default_riskFunc
    end

    csf = if haskey(kwargs, :csf)
        kwargs[:csf]
    else
        default_csf
    end

    payoffFunc = if haskey(kwargs, :payoffFunc)
        kwargs[:payoffFunc]
    else
        default_payoffFunc
    end
    @assert payoffFunc isa PayoffFunc{n}

    return Problem(d, prodFunc, riskFunc, csf, payoffFunc, costFunc)
end

"""
Handy multi-purpose constructor for `Problem` type

provide all params as keyword arguments

if `n` is not provided, it defaults to n = 2

`d` can be provided as a scalar or a vector of length `n`,
if not provided, defaults to 0

to specify the costFunc to use, provide one of the following:
- `costFunc`, which isa pre-constructed `CostFunc`
- `r0`, `r1`, and `s_thresh` as scalars or vectors of length `n` (all need to be either scalars or vectors, cannot currently mix); this will be interpreted as `CertificationCost([n,] r0, r1, s_thresh)`
- `rs` and `rp` as scalars or vectors of length `n`; this will be interpreted as `FixedUnitCost2(rs, rp)`
- `r` as a scalar or vector of length `n`; this will be interpreted as `FixedUnitCost([n,] r)`
If you provide more than one of the above, the first one in the list above will be used.
If none of the above are provided, will default to `FixedUnitCost(n, 0.1)`

to specify the prodFunc to use, provide one of the following:
- `prodFunc`, which isa pre-constructed `ProdFunc`
- arguments for `ProdFunc` constructor, e.g., one or more of (A, α, B, β, θ)
If you provide more than one of the above, the first one in the list above will be used.
If none of the above are provided, will default to `ProdFunc()`

riskFunc, csf, and payoffFunc can be provided as pre-constructed objects;
otherwise defaults `riskFunc = WinnerOnlyRisk()`, `csf = BasicCSF()`, `payoffFunc = LinearPayoff(n)` will be used
"""
function Problem(
    ;
    n::Int = 2,
    kwargs...
)
    default_d = @SVector zeros(n)
    default_CostFunc = FixedUnitCost(n, 0.1)
    default_ProdFunc = ProdFunc(; n)
    default_RiskFunc = WinnerOnlyRisk()
    default_CSF = BasicCSF()
    default_PayoffFunc = LinearPayoff(n)
    return Problem(
        default_d,
        default_CostFunc,
        default_ProdFunc,
        default_RiskFunc,
        default_CSF,
        default_PayoffFunc,
        n = n;
        kwargs...
    )
end

"""
Construct a problem from a pre-built problem, with some changes
"""
function Problem(
    problem::Problem{N};
    kwargs...
) where {N}
    Problem(
        problem.d,
        problem.costFunc,
        problem.prodFunc,
        problem.riskFunc,
        problem.csf,
        problem.payoffFunc,
        n = N;
        kwargs...
    )
end

function payoff(problem::Problem{N}, i::Int, Xs::AbstractVector, Xp::AbstractVector) where {N}
    (s, p) = problem.prodFunc(Xs, Xp)
    proba_win = problem.csf(p)  # probability that each player wins
    (pf_win, pf_lose) = problem.payoffFunc(i, p[i])  # payoffs if player i wins/loses
    payoffs_ = [(j == i) ? pf_win : pf_lose for j in 1:N]
    σis = problem.riskFunc(s)  # vector of proba(safe) conditional on each player winning
    cond_σ = proba_win .* σis
    if problem.payoffFunc isa PayoffOnDisaster && problem.payoffFunc.whogets[i]
        return sum(payoffs_ .* cond_σ) + sum((payoffs_ .- problem.d[i]) .* proba_win .* (1 .- σis)) - problem.costFunc(i, Xs, Xp)
    else
        return sum(payoffs_ .* cond_σ) - (1 - sum(cond_σ)) * problem.d[i] - problem.costFunc(i, Xs, Xp)
    end
end

function payoffs_with_s_p(problem::Problem{N}, Xs::AbstractVector, Xp::AbstractVector, s::AbstractVector, p::AbstractVector) where {N}
    proba_win = problem.csf(p)  # probability that each player wins
    # construct matrix of payoffs
    (pf_win, pf_lose) = problem.payoffFunc(p)  # payoffs if each player wins/loses
    payoffs_ = repeat(pf_lose, 1, N)
    payoffs_[diagind(payoffs_)] = pf_win
    σis = problem.riskFunc(s)  # vector of proba(safe) conditional on each player winning
    cond_σ = proba_win .* σis  # proba that i wins and outcome is safe
    if problem.payoffFunc isa PayoffOnDisaster
        safe_payoffs = vec(sum(payoffs_ .* cond_σ, dims = 2))
        cond_d = proba_win .* (1 .- σis)  # proba that i wins and outcome is not safe
        disaster_payoffs = vec(
            sum((payoffs_ .* problem.payoffFunc.whogets .- problem.d) .* cond_d,
            dims = 2
        ))
        return safe_payoffs .+ disaster_payoffs .- problem.costFunc(Xs, Xp)
    else
        return vec(sum(payoffs_ .* cond_σ, dims = 2)) .- (1 .- sum(cond_σ)) .* problem.d .- problem.costFunc(Xs, Xp)
    end
end

function payoffs(problem::Problem, Xs::AbstractVector, Xp::AbstractVector)
    (s, p) = problem.prodFunc(Xs, Xp)
    return payoffs_with_s_p(problem, Xs, Xp, s, p)
end

function s_p_σ_payoffs(problem::Problem, Xs_, Xp_)
    (s, p) = f(problem.prodFunc, Xs_, Xp_)
    # proba(safe) = sum(proba(i safe) * proba(i win))
    σ = sum(problem.riskFunc(s) .* problem.csf(p))
    payoffs_ = payoffs_with_s_p(problem, Xs_, Xp_, s, p)
    return s, p, σ, payoffs_
end

function is_symmetric(problem::Problem{N}) where {N}
    (
        is_symmetric(problem.prodFunc)
        && is_symmetric(problem.costFunc)
        && all(problem.d[1] .== problem.d[2:N])
    )
end


"""
Meant to model cases where players make decisions based on possibly incorrect beliefs about model params

It is assumed that players know each other's beliefs (though maybe nobody has correct beliefs), otherwise we end up in the crazy land of higher-order beliefs

Contains a true Problem `baseProblem`, but also a vector `beliefs` of Problems, which are the Problems that each player thinks they are dealing with
"""
struct ProblemWithBeliefs{N, T <: AbstractProblem{N}} <: AbstractProblem{N}
    baseProblem::T
    beliefs::SVector{N, T}
end

function ProblemWithBeliefs(
    baseProblem::T = Problem();
    beliefs::Union{AbstractVector{T}, Nothing} = nothing
) where {N, T <: AbstractProblem{N}}
    beliefs = if isnothing(beliefs)
        @SVector fill(baseProblem, N)
    else
        SVector{N, T}(beliefs)
    end
    return ProblemWithBeliefs(baseProblem, beliefs)
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

@to_baseProblem payoff
@to_baseProblem payoffs
@to_baseProblem payoffs_with_s_p
@to_baseProblem s_p_σ_payoffs

is_symmetric(problem::ProblemWithBeliefs) = (
    is_symmetric(problem.baseProblem)
    && is_symmetric(problem.beliefs[1])
    && all(problem.beliefs[1] == b for b in problem.beliefs[2:end])
)

function ProblemWithBeliefs(
    baseProblem,
    beliefs::AbstractVector{Dict{Symbol, Any}}
)
    return ProblemWithBeliefs(
        baseProblem,
        beliefs = [Problem(baseProblem; kwargs...) for kwargs in beliefs]
    )
end
