"""
A type that controls how much each player gets if they win or lose, based on their performance.
Subtypes must implement functions `payoff_win` and `payoff_lose`
"""
abstract type PayoffFunc end

"""
Player gets payoff a_win * p + b_win if they win, a_lose * p + b_lose if they lose,
where p is their performance.
"""
struct LinearPayoff{T <: Real} <: PayoffFunc
    a_win::T
    b_win::T
    a_lose::T
    b_lose::T
end

function LinearPayoff(; a_win::Real = 1., b_win::Real = 0., a_lose::Real = 0., b_lose::Real = 0.)
    @assert typeof(a_win) == typeof(b_win) == typeof(a_lose) == typeof(b_lose) "Params for LinearPayoff need to have same numeric type"
    return LinearPayoff(a_win, b_win, a_lose, b_lose)
end

function payoff_win(pf::LinearPayoff, p::Real)
    return pf.a_win + p * pf.b_win
end

function payoff_lose(pf::LinearPayoff, p::Real)
    return pf.a_lose + p * pf.b_lose
end