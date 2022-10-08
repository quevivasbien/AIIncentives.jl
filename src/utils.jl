# Utility functions, not dependent on special types, used throughout the package

function as_Float64_Array(x::Union{Real, AbstractArray}, n::Int)
    if isa(x, AbstractArray)
        return convert(Array{Float64}, x)
    end
    return fill(convert(Float64, x), n)
end

function mean(x, dim::Integer)
    reshape(
        sum(x, dims = dim) ./ size(x, dim),
        (s for (i, s) in enumerate(size(x)) if i != dim)...
    )
end

function mean(x, dims::Tuple)
    reshape(
        sum(x, dims = dims) ./ prod(to_indices(size(x), dims)),
        (s for (i, s) in enumerate(size(x)) if i ∉ dims)...
    )
end

function mean(x)
    sum(x) ./ length(x)
end

function to_rowvec(x::Vector)
    reshape(x, 1, :)
end

function slices_approx_equal(array, dim, atol, rtol)
    all(
        isapprox(
            selectdim(array, dim, 1), selectdim(array, dim, i),
            atol = atol, rtol = rtol
        )
        for i in 2:size(array, dim)
    )
end

function is_napprox_greater(a, b; rtol = EPSILON)
    a > b && !isapprox(a, b, rtol = rtol)
end

function get_proba(s::Real)
    proba = s / (1. + s)
    return if isnan(s) || isinf(s)
        1::typeof(proba)
    else
        proba
    end
end

function get_probas(s::AbstractVector{T}) where {T <: Real}
    probas = s ./ (1. .+ s)
    probas[isnan.(s) .| isinf.(s)] .= 1
    return probas
end
