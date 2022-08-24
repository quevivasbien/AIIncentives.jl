"""
A type that controls the probability of a safe outcome
Subtypes must implement function `σ(rf, p)` and `σ(rf, i, p)`
"""
abstract type RiskFunc end

(rf::RiskFunc)(
    s::AbstractVector{T}
) where {T <: Real} = σ(rf, s)

(rf::RiskFunc)(
    i::Int,
    s::AbstractVector{T}
) where {T <: Real} = σ(rf, i, s)

"""
Proba(safe) is weighted geometric mean of s / (1+s), to the nth power
w is vector of weights
if w is all ones, proba(safe) is simply the product of s / (1+s)
"""
struct MultiplicativeRisk{T <: Real} <: RiskFunc
    w::Vector{T}
    cum_w::T  # sum of w
    n::Int  # length of w
end

function MultiplicativeRisk(n::Int, w::T = 1.0) where {T <: Real}
    @assert n >= 2 "n must be at least 2"
    w_ = convert(Float64, w)
    return MultiplicativeRisk(fill(w_, n), w_, n)
end

function MultiplicativeRisk(w::Vector{T}) where {T <: Real}
    n = length(w)
    @assert n >= 2 "n must be at least 2"
    w_ = convert(Array{Float64}, w)
    return MultiplicativeRisk(w_, n / sum(w_), n)
end

function σ(
    rf::MultiplicativeRisk,
    i::Int,
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
    return fill(prod(probas .* rf.w) ^ rf.cum_w, length(rf.w))
end


"""
Proba(safe) is weighted arithmetic mean of s / (1+s)
w is vector of weights
if w is all ones, the proba(safe) is simply mean of s / (1+s)
"""
struct AdditiveRisk{T <: Real} <: RiskFunc
    w::Vector{T}
    cum_w::T
end

function AdditiveRisk(n::Int, w::T = 1.0) where {T <: Real}
    @assert n >= 2 "n must be at least 2"
    w_ = convert(Float64, w)
    return AdditiveRisk(fill(w_, n), n / w_)
end

function AdditiveRisk(w::Vector{T}) where {T <: Real}
    n = length(w)
    @assert n >= 2 "n must be at least 2"
    w_ = convert(Array{Float64}, w)
    return AdditiveRisk(w_, n / sum(w_))
end

function σ(
    rf::AdditiveRisk,
    i::Int,
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
    return fill(sum(probas .* rf.w) / rf.cum_w, length(rf.w))
end

"""
Proba(safe) is probability that the winner is safe
"""
struct WinnerOnlyRisk <: RiskFunc end

function σ(
    rf::WinnerOnlyRisk,
    i::Int,
    s::AbstractVector{T}
) where {T <: Real}
    return get_proba(s[i])
end

function σ(
    rf::WinnerOnlyRisk,
    s::AbstractVector{T}
) where {T <: Real}
    return get_probas(s)
end
