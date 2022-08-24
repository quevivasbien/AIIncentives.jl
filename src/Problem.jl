struct Problem{T <: Real, R <: RiskFunc, C <: CSF, P <: PayoffFunc}
    n::Int
    d::Vector{T}
    r::Vector{T}
    prodFunc::ProdFunc{T}
    riskFunc::R
    csf::C
    payoffFunc::P
end

function Problem(
    ;
    n::Int = 2,
    d::Union{Real, AbstractVector} = 0.,
    r::Union{Real, AbstractVector} = 0.1,
    riskFunc::RiskFunc = WinnerOnlyRisk(),
    prodFunc::ProdFunc{Float64} = ProdFunc(),
    csf::CSF = BasicCSF(),
    payoffFunc::PayoffFunc = LinearPayoff(0., 1., 0., 0.)
)
    @assert n >= 2 "n must be at least 2"
    @assert n == prodFunc.n "n must match prodFunc.n"
    problem = Problem(
        n,
        as_Float64_Array(d, n),
        as_Float64_Array(r, n),
        prodFunc,
        riskFunc,
        csf,
        payoffFunc
    )
    @assert all(length(getfield(problem, x)) == n for x in [:d, :r]) "Your input params need to match the number of players"
    return problem
end

function payoff(problem::Problem, i::Int, Xs::Vector, Xp::Vector)
    (s, p) = problem.prodFunc(Xs, Xp)
    proba_win = problem.csf(p)  # probability that each player wins
    pf_lose = payoff_lose(problem.payoffFunc, p[i])  # payoff if player i loses
    pf_win = payoff_win(problem.payoffFunc, p[i])  # payoff if player i wins
    payoffs = [(j == i) ? pf_win : pf_lose for j in 1:problem.n]
    σis = problem.riskFunc(s)  # vector of proba(safe) conditional on each player winning
    cond_σ = proba_win .* σis
    return sum(payoffs .* cond_σ) - (1 - sum(cond_σ)) * problem.d[i] - problem.r[i] .* (Xs[i] + Xp[i])
end

(problem::Problem)(i::Int, Xs::Vector, Xp::Vector) = payoff(problem, i, Xs, Xp)

function all_payoffs_with_s_p(problem::Problem, Xs::Vector, Xp::Vector, s::Vector, p::Vector)
    proba_win = problem.csf(p)  # probability that each player wins
    # construct matrix of payoffs
    pf_lose = payoff_lose.(Ref(problem.payoffFunc), p)  # payoff if each player loses
    pf_win = payoff_win.(Ref(problem.payoffFunc), p)  # payoff if player i wins
    payoffs = repeat(pf_lose, 1, problem.n)
    payoffs[diagind(payoffs)] = pf_win
    σis = problem.riskFunc(s)  # vector of proba(safe) conditional on each player winning
    cond_σ = proba_win .* σis
    return vec(sum(payoffs .* cond_σ, dims = 2)) .- (1 .- sum(cond_σ)) .* problem.d .- problem.r .* (Xs .+ Xp)
end

function all_payoffs(problem::Problem, Xs::Vector, Xp::Vector)
    (s, p) = problem.prodFunc(Xs, Xp)
    return all_payoffs_with_s_p(problem, Xs, Xp, s, p)
end

(problem::Problem)(Xs::Vector, Xp::Vector) = all_payoffs(problem, Xs, Xp)

function get_func(problem::Problem, i::Int, strats::Array)
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
