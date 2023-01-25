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

struct FixedUnitCost{N} <: CostFunc{N}
    r::MVector{N, Float64}
end

function FixedUnitCost(r::AbstractVector{T}) where {T <: Real}
    return FixedUnitCost(MVector{length(r), Float64}(r))
end

function FixedUnitCost(n::Int, r::Real)
    return FixedUnitCost(@MVector fill(Float64(r), n))
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
struct FixedUnitCost2{N} <: CostFunc{N}
    # r has a column for each of rs and rp
    # each row corresponds to a different player
    r::MMatrix{N, 2, Float64}
end

function FixedUnitCost2(rs::AbstractVector, rp::AbstractVector)
    n = length(rs)
    @assert n == length(rp) "rs and rp must have same length"
    return FixedUnitCost2([MVector{n, Float64}(rs) MVector{n, Float64}(rp)])
end

function FixedUnitCost2(n::Int, rs::Real, rp::Real)
    rs = @MVector fill(Float64(rs), n)
    rp = @MVector fill(Float64(rp), n)
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
struct LinearCost{N} <: CostFunc{N}
    r0::MMatrix{N, 2, Float64}
    r1::MMatrix{N, 2, Float64}
end

function LinearCost(n::Int, r0::Float64, r1::Float64)
    r0 = @MMatrix fill(r0, n, 2)
    r1 = @MMatrix fill(r1, n, 2)
    return LinearCost(r0, r1)
end

# this method sets same prices for all players, but different prices for rs and rp
# r0 and r1 should each be vectors, one entry for each of rs and rp
function LinearCost(n::Int, r0::AbstractVector{T}, r1::AbstractVector{T}) where {T <: Real}
    @assert length(r0) == 2 && length(r1) == 2 "r0 and r1 should both have length 2 (one entry for Xs, one for Xp)"
    return LinearCost(repeat(MVector{2, Float64}(r0)', n), repeat(MVector{2, Float64}(r1)', n))
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
struct CertificationCost{N} <: CostFunc{N}
    r0::MVector{N, Float64}
    r1::MVector{N, Float64}
    s_thresh::MVector{N, Float64}
    prodFunc::ProdFunc{N}
end

"""
construct CertificationCost with same r0, r1, and s_thresh for all players
"""
function CertificationCost(
    n::Int,
    r0::Real,
    r1::Real,
    s_thresh::Real,
    prodFunc::ProdFunc,
)
    return CertificationCost(
        fill(r0, MVector{n}),
        fill(r1, MVector{n}),
        fill(s_thresh, MVector{n}),
        prodFunc
    )
end

"""
construct CertificationCost with different r0, r1, and s_thresh for players
"""
function CertificationCost(
    r0::AbstractVector,
    r1::AbstractVector,
    s_thresh::AbstractVector,
    prodFunc::ProdFunc{N},
) where {N}
    @assert N == length(r0) == length(r1) == length(s_thresh) "r0, r1, and s_thresh must have same length as prodFunc's N"
    return CertificationCost(
        MVector{N, Float64}(r0),
        MVector{N, Float64}(r1),
        MVector{N, Float64}(s_thresh),
        prodFunc
    )
end

function cost(c::CertificationCost, i::Int, Xs, Xp)
    (s, _) = c.prodFunc(i, Xs[i], Xp[i])
    r = s < c.s_thresh[i] ? c.r0[i] : c.r1[i]
    return r * (Xs[i] + Xp[i])
end

function cost(c::CertificationCost, Xs, Xp)
    (s, _) = c.prodFunc(Xs, Xp)
    qualifies = s .>= c.s_thresh
    return (c.r1 .* qualifies .+ c.r0 .* (1 .- qualifies)) .* (Xs .+ Xp)
end

function is_symmetric(c::CertificationCost)
    return all(c.r0[1] .== c.r0[2:end]) && all(c.r1[1] .== c.r1[2:end]) && all(c.s_thresh[1] .== c.s_thresh[2:end])
end


"""
Same as certification cost, but discount is now applied if safety is greater than avg. of competitors' safety by some margin δ
"""
struct RelativeSafetyCertificationCost{N} <: CostFunc{N}
    r0::MVector{N, Float64}
    r1::MVector{N, Float64}
    δ::MVector{N, Float64}
    prodFunc::ProdFunc{N}
end

function cost(c::RelativeSafetyCertificationCost, i::Int, Xs, Xp)
    (s, _) = c.prodFunc(Xs, Xp)
    r = s[i] >= mean([s[1:i-1]; s[i+1:end]]) + c.δ[i] ? c.r1[i] : c.r0[i]
    return r * (Xs[i] + Xp[i])
end

function cost(c::RelativeSafetyCertificationCost, Xs, Xp)
    (s, _) = c.prodFunc(Xs, Xp)
    s_mat = repeat(s, 1, length(s)); s_mat[diagind(s_mat)] .= 0.
    qualifies = s .>= mean(s_mat, 1) .+ c.δ
    return (c.r1 .* qualifies .+ c.r0 .* (1 .- qualifies)) .* (Xs .+ Xp)
end

function is_symmetric(c::RelativeSafetyCertificationCost)
    return all(c.r0[1] .== c.r0[2:end]) && all(c.r1[1] .== c.r1[2:end]) && all(c.δ[1] .== c.δ[2:end])
end


"""
A cost schedule where players pay a fixed unit cost r, plus a penalty δ if their safety is below a threshold `s_thresh`
"""
struct PenaltyCost{N} <: CostFunc{N}
    r::MVector{N, Float64}
    δ::MVector{N, Float64}
    s_thresh::MVector{N, Float64}
    prodFunc::ProdFunc{N}
end

"""
construct PenaltyCost with same values for all players
"""
function PenaltyCost(
    n::Int,
    r::Real,
    δ::Real,
    s_thresh::Real,
    prodFunc::ProdFunc,
)
    return PenaltyCost(
        fill(r, MVector{n}),
        fill(δ, MVector{n}),
        fill(s_thresh, MVector{n}),
        prodFunc
    )
end

"""
construct PenaltyCost with different values for players
"""
function PenaltyCost(
    r::AbstractVector,
    δ::AbstractVector,
    s_thresh::AbstractVector,
    prodFunc::ProdFunc{N},
) where {N}
    @assert N == length(r) == length(δ) == length(s_thresh) "r, δ, and s_thresh must have same length as prodFunc's N"
    return PenaltyCost(
        MVector{N, Float64}(r),
        MVector{N, Float64}(δ),
        MVector{N, Float64}(s_thresh),
        prodFunc
    )
end

function cost(c::PenaltyCost, i::Int, Xs, Xp)
    (s, _) = c.prodFunc(i, Xs[i], Xp[i])
    penalty = s < c.s_thresh[i] ? c.δ[i] : 0
    return c.r[i] * (Xs[i] + Xp[i]) + penalty
end

function cost(c::PenaltyCost, Xs, Xp)
    (s, _) = c.prodFunc(Xs, Xp)
    gets_penalty = s .< c.s_thresh
    return c.r .* (Xs .+ Xp) .+ c.δ .* gets_penalty
end

function is_symmetric(c::PenaltyCost)
    return all(c.r[1] .== c.r[2:end]) && all(c.δ[1] .== c.δ[2:end]) && all(c.s_thresh[1] .== c.s_thresh[2:end])
end
