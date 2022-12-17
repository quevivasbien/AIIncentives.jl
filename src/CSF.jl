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

# default impl of proba_win(csf, p) if only proba_win(csf, i, p) is defined
function proba_win(csf::CSF, p::AbstractVector)
    return proba_win.(Ref(csf), 1:length(p), Ref(p))
end


"Probability of i winning is p[i] / sum(p)"
struct BasicCSF <: CSF end

function proba_win(::BasicCSF, p::AbstractVector)
    return p ./ sum(p)
end

function proba_win(::BasicCSF, i::Int, p::AbstractVector)
    return p[i] / sum(p)
end


"""
Probability of i winning is w * p[i] / (1 + w * sum(p))
Idea is that proba that someone wins is w * sum(p) / (1 + w * sum(p)),
    and proba that i wins given someone wins is p[i] / sum(p)
"""
Base.@kwdef struct MaybeNoWinCSF <: CSF
    w::Float64 = 1.0
end

function proba_win(csf::MaybeNoWinCSF, p::AbstractVector)
    return csf.w .* p ./ (1. + csf.w * sum(p))
end

function proba_win(csf::MaybeNoWinCSF, i, p::AbstractVector)
    return csf.w * p[i] / (1. + csf.w * sum(p))
end
