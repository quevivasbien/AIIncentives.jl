"""
Determines cost of inputs
Subtypes should implement `cost` and `is_symmetric`
"""
abstract type CostFunc{N} end

function (c::CostFunc)(i::Int, Xs::AbstractVector, Xp::AbstractVector)
    return cost(c, i, Xs, Xp)
end

function (c::CostFunc)(Xs::AbstractVector, Xp::AbstractVector)
    return cost(c, Xs, Xp)
end

# default impl if not implemented manually
function cost(c::CostFunc{N}, Xs, Xp) where {N}
    return cost.(Ref(c), 1:N, Ref(Xs), Ref(Xp))
end

mutable struct FixedUnitCost{N} <: CostFunc{N}
    r::SVector{N, Float64}
end

function FixedUnitCost(r::AbstractVector{T}) where {T <: Real}
    return FixedUnitCost(SVector{length(r), Float64}(r))
end

function FixedUnitCost(n::Int, r::Real)
    return FixedUnitCost(as_SVector(r, n))
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
mutable struct FixedUnitCost2{N} <: CostFunc{N}
    # r has a column for each of rs and rp
    # each row corresponds to a different player
    r::SMatrix{N, 2, Float64}
end

function FixedUnitCost2(rs::AbstractVector, rp::AbstractVector)
    n = length(rs)
    @assert n == length(rp) "rs and rp must have same length"
    return FixedUnitCost2([as_SVector(rs, n) as_SVector(rp, n)])
end

function FixedUnitCost2(n::Int, rs::Real, rp::Real)
    rs = @SVector fill(Float64(rs), n)
    rp = @SVector fill(Float64(rp), n)
    return FixedUnitCost2([rs rp])
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
mutable struct LinearCost{N} <: CostFunc{N}
    r0::SMatrix{N, 2, Float64}
    r1::SMatrix{N, 2, Float64}
end

function LinearCost(n::Int, r0::Float64, r1::Float64)
    r0 = @SMatrix fill(r0, n, 2)
    r1 = @SMatrix fill(r1, n, 2)
    return LinearCost(r0, r1)
end

# this method sets same prices for all players, but different prices for rs and rp
# r0 and r1 should each be vectors, one entry for each of rs and rp
function LinearCost(n::Int, r0::AbstractVector{T}, r1::AbstractVector{T}) where {T <: Real}
    @assert length(r0) == 2 && length(r1) == 2 "r0 and r1 should both have length 2 (one entry for Xs, one for Xp)"
    return LinearCost(n, repeat(r0', n), repeat(r1', n))
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


"""
A cost schedule where players pay a fixed unit cost r0 if their safety is less than a threshold `s_thresh` and a fixed unit cost r1 otherwise
"""
mutable struct CertificationCost{N} <: CostFunc{N}
    r0::SVector{N, Float64}
    r1::SVector{N, Float64}
    s_thresh::SVector{N, Float64}
    const prodFunc::ProdFunc{N}
end

"""
construct CertificationCost with same r0 and r1 for all players
"""
function CertificationCost(
    n::Int,
    r0::Real,
    r1::Real,
    s_thresh::Real,
    prodFunc::ProdFunc,
)
    return CertificationCost(
        as_SVector(r0, n),
        as_SVector(r1, n),
        as_SVector(s_thresh, n),
        prodFunc
    )
end

"""
construct CertificationCost with different r0 and r1 for players
"""
function CertificationCost(
    r0::AbstractVector,
    r1::AbstractVector,
    s_thresh::AbstractVector,
    prodFunc::ProdFunc{N},
) where {N}
    @assert N == length(r0) == length(r1) == length(s_thresh) "r0, r1, and s_thresh must have same length as prodFunc's N"
    return CertificationCost(
        SVector{N, Float64}(r0),
        SVector{N, Float64}(r1),
        SVector{N, Float64}(s_thresh),
        prodFunc
    )
end

function cost(c::CertificationCost, i::Int, Xs, Xp)
    (s, _) = c.prodFunc(i, Xs[i], Xp[i])
    r = s < c.s_thresh[i] ? c.r0[i] : c.r1[i]
    return r * (Xs[i] + Xp[i])
end

function is_symmetric(c::CertificationCost)
    return all(c.r0[1] .== c.r0[2:end]) && all(c.r1[1] .== c.r1[2:end])
end
