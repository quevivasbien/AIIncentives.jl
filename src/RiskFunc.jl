"""
A type that controls the probability of a safe outcome
Subtypes must implement functions `σ(rf, p)` and `σ(rf, i, p)`
"""
abstract type RiskFunc end


"Returns vector of probabilies: proba(safe) conditional on each player winning"
(rf::RiskFunc)(
    s::AbstractVector{T}
) where {T <: Real} = σ(rf, s)

"Returns proba(safe) conditional on player i winning"
(rf::RiskFunc)(
    i::Int,
    s::AbstractVector{T}
) where {T <: Real} = σ(rf, i, s)

"""
Defines proba(safe) as weighted geometric mean of s / (1+s), to the nth power
w is vector of weights
if w is all ones, proba(safe) is simply the product of s / (1+s)
"""
struct MultiplicativeRisk{N} <: RiskFunc
    w::SVector{N, Float64}
    cum_w::Float64  # n / sum(w)
end

function MultiplicativeRisk(n::Int, w::T = 1.0) where {T <: Real}
    @assert n >= 2 "n must be at least 2"
    w_ = as_SVector(w, n)
    return MultiplicativeRisk(w_, 1. / w)
end

function MultiplicativeRisk(w::AbstractVector{T}) where {T <: Real}
    n = length(w)
    @assert n >= 2 "n must be at least 2"
    w_ = SVector{n, Float64}(w)
    return MultiplicativeRisk(w_, n / sum(w_), n)
end

function σ(
    rf::MultiplicativeRisk,
    ::Int,
    s::AbstractVector{T}
) where {T <: Real}
    probas = get_probas(s)
    return prod(probas .* rf.w) ^ rf.cum_w
end

function σ(
    rf::MultiplicativeRisk,
    s::AbstractVector{T}
) where {T <: Real}
    probas = get_probas(s)
    return @SVector fill(prod(probas .* rf.w) ^ rf.cum_w, length(rf.w))
end


"""
Defines proba(safe) as weighted arithmetic sum of s / (1+s)
w is vector of weights
if w is all ones, the proba(safe) is simply sum of s / (1+s)
"""
struct AdditiveRisk{N} <: RiskFunc
    w::SVector{N, Float64}
    cum_w::T
end

function AdditiveRisk(n::Int, w::T = 1.0) where {T <: Real}
    @assert n >= 2 "n must be at least 2"
    w_ = as_SVector(w, n)
    return AdditiveRisk(w_, 1. / w)
end

function AdditiveRisk(w::AbstractVector{T}) where {T <: Real}
    n = length(w)
    @assert n >= 2 "n must be at least 2"
    w_ = SVector{n, Float64}(w)
    return AdditiveRisk(w_, n / sum(w_))
end

function σ(
    rf::AdditiveRisk,
    ::Int,
    s::AbstractVector{T}
) where {T <: Real}
    probas = get_probas(s)
    return sum(probas .* rf.w) / rf.cum_w
end

function σ(
    rf::AdditiveRisk,
    s::AbstractVector{T}
) where {T <: Real}
    probas = get_probas(s)
    return @SVector fill(sum(probas .* rf.w) / rf.cum_w, length(rf.w))
end

"""
Defines proba(safe) as probability that the winner is safe
"""
struct WinnerOnlyRisk <: RiskFunc end

function σ(
    ::WinnerOnlyRisk,
    i::Int,
    s::AbstractVector{T}
) where {T <: Real}
    return get_proba(s[i])
end

function σ(
    ::WinnerOnlyRisk,
    s::AbstractVector{T}
) where {T <: Real}
    return get_probas(s)
end
