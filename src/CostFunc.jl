abstract type CostFunc end

function (c::CostFunc)(i::Int, Xs, Xp)
    cost(c, i, Xs, Xp)
end

function (c::CostFunc)(Xs, Xp)
    cost(c, Xs, Xp)
end

struct FixedUnitCost{T <: Real} <: CostFunc
    r::Vector{T}
end

function FixedUnitCost(n::Int, r::T) where T <: Real
    return FixedUnitCost(fill(r, n))
end

function cost(c::FixedUnitCost, i::Int, Xs, Xp)
    return c.r[i] * (Xs[i] + Xp[i])
end

function cost(c::FixedUnitCost, Xs, Xp)
    return c.r .* (Xs .+ Xp)
end
