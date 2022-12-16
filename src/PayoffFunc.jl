"""
A type that controls how much each player gets if they win or lose, based on their performance.
Subtypes must implement functions `payoff_win` and `payoff_lose`
"""
abstract type PayoffFunc{N} end

(pf::PayoffFunc)(i, p::Real) = (payoff_win(pf, i, p), payoff_lose(pf, i, p))
(pf::PayoffFunc)(p::AbstractVector) = (payoff_win(pf, p), payoff_lose(pf, p))

"""
Player gets payoff a_win * p + b_win if they win, a_lose * p + b_lose if they lose,
where p is their performance.
"""
struct LinearPayoff{N} <: PayoffFunc{N}
    a_win::SVector{N, Float64}
    b_win::SVector{N, Float64}
    a_lose::SVector{N, Float64}
    b_lose::SVector{N, Float64}
end

function LinearPayoff(n = 2; a_win = 1., b_win = 0., a_lose = 0., b_lose = 0.)
    linearPayoff = LinearPayoff(
        as_SVector(a_win, n),
        as_SVector(b_win, n),
        as_SVector(a_lose, n),
        as_SVector(b_lose, n)
    )
    return linearPayoff
end

function payoff_win(pf::LinearPayoff, p::AbstractVector)
    return pf.a_win .+ p .* pf.b_win
end

function payoff_win(pf::LinearPayoff, i::Int, p::Real)
    return pf.a_win[i] + p * pf.b_win[i]
end

function payoff_lose(pf::LinearPayoff, p::AbstractVector)
    return pf.a_lose .+ p .* pf.b_lose
end

function payoff_lose(pf::LinearPayoff, i::Int, p::Real)
    return pf.a_lose[i] + p * pf.b_lose[i]
end

"""
Allow for some or all players to get payoff even if a disaster occurs
basePayoff is another PayoffFunc that determines payoffs
whogets is a vector of length n that indicates who can get a payoff if there's a disaster
"""
struct PayoffOnDisaster{N, T <: PayoffFunc{N}} <: PayoffFunc{N}
    basePayoff::T
    whogets::SVector{N, Bool}
end

function PayoffOnDisaster(
    basePayoff::PayoffFunc{N} = LinearPayoff()
    ;
    whogets::Union{AbstractVector{Bool}, Nothing} = nothing
) where {N}
    if isnothing(whogets)
        return PayoffOnDisaster(
            basePayoff,
            @SVector fill(true, n)  # default is all players can get payoff even if disaster occurs
        )
    else
        @assert length(whogets) == n "length of indicator vector `whogets` must match basePayoff.n"
        return PayoffOnDisaster(basePayoff, SVector{N, Bool}(whogets))
    end
end

# forward calls to basePayoff by default
macro to_basePayoff(f)
    fname = Symbol(f)
    quote
        function $(esc(fname))(pf::PayoffOnDisaster, args...)
            return $(esc(fname))(pf.basePayoff, args...)
        end
    end
end

@to_basePayoff payoff_win
@to_basePayoff payoff_lose
