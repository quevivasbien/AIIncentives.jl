# Utility functions, not dependent on special types, used throughout the package

function as_Float64_Array(x::Union{Real, AbstractArray}, n::Int)
    if isa(x, AbstractArray)
        @assert length(x) == n
        return convert(Array{Float64}, x)
    end
    return fill(convert(Float64, x), n)
end


# Helpers for combining arrays

# stack arrays along a new dimension
function stack(arrays...; dim::Integer = 1)
    n = ndims(arrays[1])
    @assert all(ndims(arrays[1]) == n for a in arrays[2:end])
    return permutedims(cat(arrays..., dims = n+1), (1:dim-1..., n+1, dim:n...))
end

# recursively combine vectors of vectors [of vectors...] into a single ndarray
function combine(v::Vector{T}) where {T <: AbstractVector}
    return stack([combine(vv) for vv in v]...)
end

function combine(v::Vector{T}) where {T <: Number}
    return v
end


# Some helpers for computing statistics on arrays over dimensions

function broadcast_sum(x, dim::Integer)
    reshape(
        sum(x, dims = dim),
        (s for (i, s) in enumerate(size(x)) if i!= dim)...
    )
end

function broadcast_sum(x, dims::Tuple)
    reshape(
        sum(x, dims = dims),
        (s for (i, s) in enumerate(size(x)) if i ∉ dims)...
    )
end

function ax_size(x, dim::Integer)
    return size(x, dim)
end

function ax_size(x, dims::Tuple)
    return prod(size(x)[collect(dims)])
end

function mean(x, dims)
    broadcast_sum(x, dims) ./ ax_size(x, dims)
end

function mean(x)
    sum(x) / length(x)
end

function std(x, dims)
    m = mean(x, dims)
    # put the dims at the end so we can automatically broadcast with m
    x_ = PermutedDimsArray(x, (setdiff(1:ndims(x), dims)..., dims...))
    newdims = dims isa Integer ? ndims(x) : Tuple((ndims(x) - length(dims) + 1):ndims(x))
    return sqrt.(broadcast_sum((x_ .- m) .^ 2, newdims) ./ (ax_size(x, dims) - 1))
end

function std(x)
    m = mean(x)
    return sqrt(sum((x .- m) .^ 2) / (length(x) - 1))
end

function to_rowvec(x::Vector)
    reshape(x, 1, :)
end


# checks if all slices of `array` along dimension `dim` are approximately equal
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


# convert from odds safe to proba safe, robust to nan & inf
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
