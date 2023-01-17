# Tips for testing claims

## Using `solve_left_right`

You can solve and compare lots of problems without much effort on your part using the `solve_left_right` function defined in `utils.jl`. (You may need to call it as `AIIncentives.solve_left_right` since, as of writing, it's not exported globally.)

You should use this when you have two types of problems you want to compare, and you want to compare those problems over many different parameterizations; for example, you could compare cases where $A < 10$ with cases where $A > 10$.

The function signature is
```julia
function solve_left_right(
    # values in these dicts will be unique to respective problems
    left::Dict,
    right::Dict;
    # values here will be common to both left and right problems
    common::Dict = Dict(),
    # key word args to forward to `solve`
    solver_kwargs...
)
```

The required arguments are dictionaries `left` and `right`. These define the two types of problems we want to compare. In our example with low and high values of $A$, we could define
```julia
left = Dict(:A => range(0.1, 10., length = 20))
right = Dict(:A => range(10., 100., length = 20))
```
To override the default values of other parameters, we can supply them to the `common` dictionary; e.g., to set $\theta = 2$:
```julia
common = Dict(:θ => [2.])
```
(Notice that all the values we provide to these dictionaries need to be list-like (`AbstractVector`) types. We have to say `:θ => [2.]` instead of `:θ => 2.`.)
We can also use the `common` dictionary if we want to try some values with both the left and right problems, but try the same values for both problems; e.g., to try a range of θ values:
```julia
common = Dict(:θ => range(0., 2., length = 5))
```

To put this all together, we could run
```julia
solve_left_right(
    Dict(:A => range(0.1, 10., length = 20)),
    Dict(:A => range(10., 100., length = 20)),
    common = Dict(:θ => range(0., 2., length = 5))
)
```
which will solve a bunch of problems with $A$ from 0.1 to 10 and $\theta$ from 0 to 2, and a bunch of problems with $A$ from 10 to 100 and $\theta$ again from 0 to 2.

We can specify more than just one parameter in each dictionary. For example, we could do something like
```julia
solve_left_right(
    Dict(
        :A => range(0.1, 10., length = 20),
        :B => range(0.1, 10., length = 20),
    ),
    Dict(
        :A => range(10., 100., length = 20),
        :B => range(10., 100., length = 20),
    ),
    common = Dict(:θ => range(0., 2., length = 5))
)
```
which will combine a bunch of low $A$ and $B$ values with a bunch of high $A$ and $B$ values (and the same $\theta$ values).

The values are combined as their cartesian product. This means that in the example above that the set of left-side problems will be every $A$ value in the 0.1 to 10 range combined with every $B$ value in the 10 to 100 range, in turn combined with every $\theta$ value in the 0 to 2 range; since these have 20, 20, and 5 elements, respectively, we'd have a total of 20 x 20 x 5 = 2000 problems.

The `solve_left_right` function returns an n-dimensional array of left-side results and an n-dimensional array of right-side results, where the dimension `n` of results is the total number of parameters we specified. In the example above, we'd get a pair of 20 x 20 x 5 arrays of `SolverResult`s.

We could then use these results to test hypotheses we may have. For example, suppose the hypothesis we're trying to test is, "Problems with $A \leq 10$ have lower safety than problems with $A \geq 10$, regardless of the value of $\theta$." We could test this with
```julia
l, r = solve_left_right(
    Dict(:A => range(0.1, 10., length = 20)),
    Dict(:A => range(10., 100., length = 20)),
    common = Dict(:θ => range(0., 2., length = 5))
)
# extract safety value from all results
s_l = map(x -> x.s, r)
s_r = map(x -> x.s, r)
# check that values in s_l are always no greater than values in s_r
all(all(x .<= s_r) for x in s_l)  # should be true
```