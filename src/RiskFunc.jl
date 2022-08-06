abstract type RiskFunc end


struct MultiplicativeRiskFunc{T <: Real} <: RiskFunc
    w::Vector{T}
    cum_w::T
end

function MultiplicativeRiskFunc(n::Integer, w::T = 1.0) where {T <: Real}
    @assert n >= 2 "n must be at least 2"
    w_ = convert(Float64, w)
    return MultiplicativeRiskFunc(fill(w_, n), w_)
end

function MultiplicativeRiskFunc(w::Vector{T}) where {T <: Real}
    n = length(w)
    @assert n >= 2 "n must be at least 2"
    w_ = convert(Array{Float64}, w)
    return MultiplicativeRiskFunc(w_, n / sum(w_))
end

function get_total_safety(
    rf::MultiplicativeRiskFunc,
    s::AbstractVector{T},
    p::Union{AbstractVector{T}, Nothing} = nothing
) where {T <: Real}
    # safety is weighted geometric mean of s / (1+s), to the nth power
    # if w is all ones, this is just the product of s / (1+s)
    probas = get_probas(s)
    return prod(probas .* rf.w) ^ rf.cum_w
end


struct AdditiveRiskFunc{T <: Real} <: RiskFunc
    w::Vector{T}
    cum_w::T
end

function AdditiveRiskFunc(n::Integer, w::T = 1.0) where {T <: Real}
    @assert n >= 2 "n must be at least 2"
    w_ = convert(Float64, w)
    return AdditiveRiskFunc(fill(w_, n), n / w_)
end

function AdditiveRiskFunc(w::Vector{T}) where {T <: Real}
    n = length(w)
    @assert n >= 2 "n must be at least 2"
    w_ = convert(Array{Float64}, w)
    return AdditiveRiskFunc(w_, n / sum(w_))
end

function get_total_safety(
    rf::AdditiveRiskFunc,
    s::AbstractVector{T},
    p::Union{AbstractVector{T}, Nothing} = nothing
) where {T <: Real}
    # safety is weighted arithmetic meand of s / (1+s), times n
    # if w is all ones, this is just the sum of s / (1+s)
    probas = get_probas(s)
    return sum(probas .* rf.w) * rf.cum_w
end


struct WinnerOnlyRiskFunc <: RiskFunc end

function get_total_safety(
    rf::WinnerOnlyRiskFunc,
    s::AbstractVector{T},
    p::Union{AbstractVector{T}, Nothing} = nothing
) where {T <: Real}
    # safety is the winner's probability
    probas = get_probas(s)
    return sum(p .* probas) / sum(p)
end
