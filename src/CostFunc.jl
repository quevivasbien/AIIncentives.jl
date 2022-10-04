"""
Determines cost of inputs
Subtypes should implement `cost` and `is_symmetric`
"""
abstract type CostFunc end

function (c::CostFunc)(i::Int, Xs::AbstractVector, Xp::AbstractVector)
    return cost(c, i, Xs, Xp)
end

function (c::CostFunc)(Xs::AbstractVector, Xp::AbstractVector)
    return cost(c, Xs, Xp)
end

struct FixedUnitCost{T <: Real} <: CostFunc
    n::Int
    r::Vector{T}
end

function FixedUnitCost(n::Int, r::T) where T <: Real
    return FixedUnitCost(n, fill(r, n))
end

function cost(c::FixedUnitCost, i::Int, Xs, Xp)
    return c.r[i] * (Xs[i] + Xp[i])
end

function cost(c::FixedUnitCost, Xs, Xp)
    return c.r .* (Xs .+ Xp)
end

function is_symmetric(c::FixedUnitCost)
    return all(c.r[1] .== c.r[2:end])
end


"""
Like FixedUnitCost but with different prices for Xs and Xp
"""
struct FixedUnitCost2{T <: Real} <: CostFunc
    n::Int
    r::Matrix{T}
end

function FixedUnitCost2(n::Int, rs::T, rp::T) where {T <: Real}
    return FixedUnitCost2(n, repeat([rs rp], n))
end

function cost(c::FixedUnitCost2, i::Int, Xs, Xp)
    return c.r[i, 1] * Xs[i] + c.r[i, 2] * Xp[i]
end

function cost(c::FixedUnitCost2, Xs, Xp)
    return c.r[:, 1] .* Xs .+ c.r[:, 2] .* Xp
end

function is_symmetric(c::FixedUnitCost2)
    return all(c.r[1, :]' .== c.r[2:end, :])
end


"""
Costs change linearly in Xs, Xp
Marginal cost of (Xs, Xp) is r0[1] + r1[1] * Xs + r0[2] + r1[2] * Xp
=> Total cost is r0[1] * Xs + r1[1] * Xs^2 / 2 + r0[2] * Xp + r1[2] * Xp^2 / 2
"""
struct LinearCost{T <: Real} <: CostFunc
    n::Int
    r0::Matrix{T}
    r1::Matrix{T}
end

function LinearCost(n::Int, r0::T, r1::T) where {T <: Real}
    return LinearCost(n, fill(r0, n, 2), fill(r1, n, 2))
end

function LinearCost(n::Int, r0::Vector{T}, r1::Vector{T}) where {T <: Real}
    @assert length(r0) == 2 && length(r1) == 2 "r0 and r1 should both have length 2 (one entry for Xs, one for Xp)"
    return LinearCost(n, repeat(r0', 2), repeat(r1', 2))
end

function cost(c::LinearCost, i::Int, Xs, Xp)
    return c.r0[i, 1] * Xs[i] + c.r1[i, 1] * Xs[i]^2 / 2 + c.r0[i, 2] * Xp[i] + c.r1[i, 2] * Xp[i]^2 / 2
end

function cost(c::LinearCost, Xs, Xp)
    return c.r0[:, 1] .* Xs .+ c.r1[:, 1] .* Xs.^2 ./ 2 .+ c.r0[:, 2] .* Xp .+ c.r1[:, 2] .* Xp.^2 ./ 2
end

function is_symmetric(c::LinearCost)
    return all(c.r0[1, :]' .== c.r0[2:end, :]) && all(c.r1[1, :]' .== c.r1[2:end, :])
end
