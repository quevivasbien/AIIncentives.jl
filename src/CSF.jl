"""
Controls probability that each player wins, as function of p
Subtypes must implement `proba_win(p)` and `proba_win(i, p)`
"""
abstract type CSF end

"Returns vector of probabilies that each player wins"
(csf::CSF)(
    p::AbstractVector{T}
) where {T <: Real} = proba_win(csf, p)

"Returns proba that player i wins"
(csf::CSF)(
    i::Int,
    p::AbstractVector{T}
) where {T <: Real} = proba_win(csf, i, p)


"Probability of i winning is p[i] / sum(p)"
struct BasicCSF <: CSF end

function proba_win(csf::BasicCSF, p::AbstractVector{T}) where {T <: Real}
    return p ./ sum(p)
end

function proba_win(csf::BasicCSF, i::Int, p::AbstractVector{T}) where {T <: Real}
    return p[i] / sum(p)
end
