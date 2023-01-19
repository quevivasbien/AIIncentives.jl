# Utility functions, not dependent on special types, used throughout the package

# Helpers for coercing to Array and SArray types

function as_Float64_Array(x::Real, n::Int)
    return fill(Float64(x), n)
end

function as_Float64_Array(x::AbstractArray, n::Int)
    @assert length(x) == n
    return convert(Array{Float64}, x)
end

function as_SVector(x::Real, n::Int)
    x = Float64(x)
    return @SVector fill(x, n)
end

function as_SVector(x::AbstractVector, n::Int)
    return SVector{n, Float64}(x)
end


# Helpers for combining arrays

# stack arrays along a new dimension
function stack(arrays...; dim::Integer = 1)
    n = ndims(arrays[1])
    @assert all(ndims(arrays[1]) == n for a in arrays[2:end])
    return permutedims(cat(arrays..., dims = n+1), (1:dim-1..., n+1, dim:n...))
end

# recursively combine vectors of vectors [of vectors...] into a single ndarray
function combine(v::AbstractVector{T}) where {T <: AbstractVector}
    return stack([combine(vv) for vv in v]...)
end

function combine(v::AbstractVector{T}) where {T <: Number}
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
function get_proba(s::Float64)
    proba = s / (1. + s)
    return if isnan(s) || isinf(s)
        1.
    else
        proba
    end
end


# functions to help explore claims

# helper for solve_left_right; combines two dicts of params and forms problems from their cartesian product
function combine_params(params, base_params)
    params = merge(params, base_params)
    kwargs = [Dict(zip(keys(params), p)) for p in Iterators.product(values(params)...)]
    problems = Array{Problem}(undef, size(kwargs))
    Threads.@threads for idx in eachindex(kwargs)
        problems[idx] = Problem(; kwargs[idx]...)
    end
    return problems
end

# helper for solve_left_right; solves a set of problems with params from params and base_params
function solve_side(
    params,
    base_params;
    solver_kwargs...
)
    problems = combine_params(params, base_params)
    println("$(length(problems)) problems to solve...")
    solutions = Array{SolverResult}(undef, size(problems))
    time0 = time()
    Threads.@threads for idx in eachindex(problems)
        solutions[idx] = solve(problems[idx]; solver_kwargs...)
    end
    println("finished in $(time() - time0) seconds ($(1000 * (time() - time0) / length(problems)) ms per problem)")
    return solutions
end

# solve two sets of problems `left` and `right` with different sets of assumptions
function solve_left_right(
    # values in these dicts will be unique to respective problems
    left::Dict,
    right::Dict;
    # values here will be common to both left and right problems
    common::Dict = Dict(),
    # key word args to forward to `solve`
    solver_kwargs...
)
    # finds solutions over all combinations of parameters
    println("solving left side...")
    left_sols = solve_side(left, common; solver_kwargs...)
    println("solving right side...")
    right_sols = solve_side(right, common; solver_kwargs...)
    return left_sols, right_sols
end
