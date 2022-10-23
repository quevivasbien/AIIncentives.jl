# AIIncentives.jl

## What is this?

The code in this repository is meant to find Nash equilibria for the following model:

We assume that *n* players produce safety, *s*, and performance, *p*, as

$$s_i = A_i X_{s,i}^{\alpha_i} p_i^{-\theta_i}, \quad p_i = B_i X_{p,i}^{\beta_i}$$

for *i = 1, ..., n*. The *X* are inputs chosen by the players, and all other variables are fixed parameters.

In a Nash equilibrium, each player *i* chooses *X<sub>s,i</sub>* and *X<sub>p,i</sub>* to maximize the payoff

$$u_i := \sum_{j=1}^n \sigma_j(s) q_j(p) \rho_{ij}(p) - \left( 1 - \sum_{j=1}^n \sigma_j(s) q_j(p) \right) d_i - c_i(X_s, X_p)$$

subject to the other players' choices of *X<sub>s</sub>* and *X<sub>p</sub>*. The components of this expression will be explained more below, but the basic parts are as follows:

* *q<sub>i</sub>(p)* is the probability that player *i* wins a contest between all players.
* *&sigma;<sub>i</sub>(s)* is the probability of a safe outcome given that player *i* wins the contest.
* *&rho;<sub>ij</sub>(p)* is player *i*'s payoff if player *j* wins the contest, and the outcome is safe.
* *d<sub>i</sub>* is the cost incurred by player *i* in the event of an unsafe (disaster) outcome.
* *c<sub>i</sub>(X<sub>s</sub>, X<sub>p</sub>)* is the price that player *i* pays for the inputs.

Note that the weighted sum *&Sigma;<sub>i</sub> &sigma;<sub>i</sub> q<sub>i</sub>* is the unconditional probability of a safe outcome (no disaster). 

## Getting started

If you don't have Julia, download it from https://julialang.org/downloads/ and install. (You'll need at least version 1.6.7.)

Next, you'll need to download the code from this project's GitHub repository (if you haven't already). Save it wherever you want on your computer. It's best to use git, or the GitHub CLI or desktop app, to do this so it's easy to stay in synch with the latest version.

There are at this point two ways to import the project code: 

### Method 1

At this point, the easiest way to load the project code is to open a new Julia session in the project directory -- from your computer's terminal, navigate to the project directory and run:
```bash
julia --project --threads=auto
```
or, from an arbitrary directory:
```bash
julia --project=/path/to/AIIncentives.jl --threads=auto
```

(The `--project` flag allows you to import the project with the `using` keyword, and the `--threads=auto` flag allows Julia to use all of your computer's CPU cores, which can speed some tasks up significantly.)

The first time you do this, you will probably need to run
```
] resolve
```
to get the package set up (including installing any needed dependencies).

Then, to load the project code, just run
```
using AIIncentives
```
in your new Julia session. If this is the first time you do this, there will be a bit of delay as it precompiles some of the code.

### Method 2

As an alternative, you can also use the code by launching Julia without specifying a project directory -- e.g., `julia --threads=auto` -- then including `src/includes.jl` -- e.g., from the project directory, `include("src/includes.jl")`. If you do this, you'll need to make sure you have installed the project's dependencies beforehand. I.e., something like the following should work:

In a terminal from the project directory:
```bash
julia --threads=auto
```

Then, in the Julia REPL:
```
] add Optim, NLsolve, Plots
include("src/includes.jl")
```

You will only need to `] add` the dependencies the first time you run this. Subsequently, you will just need to `include` the project code.

(This latter method is better if you're messing around with the project's source code, but otherwise means you will need to recompile more code every time you use it.)

---

You can then create and solve a scenario like
```julia
scenario = Scenario(
    n = 2,
    A = 10.,
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = 0.25, 
    d = 1.,
    r = range(0.01, 0.1, length = 20),
    varying = :r
)

solution = solve(scenario)
```
[Tip: to use math-related characters in most Julia code, you can typically just type the Latex code then press Tab. For example, `\alpha` + `[Tab]` becomes `α`. I've also allowed the spelled-out versions of Greek letters to work in most places in this code; for example, you could use `alpha`, `beta`, and `theta` instead of `α`, `β`, and `θ` in the example above.]

This will find Nash equilibria over the given parameterization and range of values for `r`. To generate a plot of the result, you can execute
```julia
plot(solution)
```

The first time you run something in the Julia session, it will take a while, since Julia compiles your code the first time you run it in a new setting. It should run a lot faster after that.

If you just want to solve and plot scenarios where some parameter value changes, you can skip now to the section on the `Scenario` type. Otherwise, if you want to understand the code at a more fundamental level, keep reading in a linear fashion.

## Base types

### The `ProdFunc` type

The `ProdFunc.jl` file implements a `ProdFunc` (production function) type that contains the variables for the function

$$f(X_s, X_p) = (AX_s^\alpha (BX_p^\beta)^{-\theta},\ BX_p^\beta)$$

describing how the inputs *X<sub>s</sub>* and *X<sub>p</sub>* produce safety and performance for all players. You can create an instance of the `ProdFunc` type like
```julia
prodFunc = ProdFunc(
    n = 2,
    A = 10.,
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = 0.25
)
```
If you want to supply different parameters for each player or just like typing more stuff out, you can also supply the parameters as vectors of equal length (equal to `n`). The following creates the same object as the example above:
```julia
prodFunc = ProdFunc(
    n = 2,
    A = [10., 10.],
    α = [0.5, 0.5],
    B = [10., 10.],
    β = [0.5, 0.5],
    θ = [0.25, 0.25]
)
```
If you omit a keyword argument, it will be set at a default value.

[Tip: in general, if you want to see the different ways you can call something, you can use Julia's handy `methods` function. For example, to see the constructors available for `ProdFunc`, you can call `methods(ProdFunc)`, which will return something like:
```
# 3 methods for type constructor:
[1] ProdFunc(; n, A, α, B, β, θ) in Main at ~/Documents/code/AIIncentives.jl/src/ProdFunc.jl:57
[2] ProdFunc(n::Int64, A::Vector{T}, α::Vector{T}, B::Vector{T}, β::Vector{T}, θ::Vector{T}) where T<:Real in Main at ~/Documents/code/AIIncentives.jl/src/ProdFunc.jl:49
[3] ProdFunc(A, α, B, β, θ) in Main at ~/Documents/code/AIIncentives.jl/src/ProdFunc.jl:79
```
This tells us that we can make a new `ProdFunc` by [1] providing keyword arguments, like in the examples shown above (anything after a semicolon in a Julia function definition is a keyword argument), or [2]-[3] providing all the required arguments in order.]

To determine the outputs for all players, you can do
```julia
(s, p) = f(prodFunc, Xs, Xp)
```
or, equivalently,
```julia
(s, p) = prodFunc(Xs, Xp)
```
where `Xs` and `Xp` are vectors of length equal to `prodFunc.n`. Here `s` and `p` will be vectors representing the safety and performance of each player. To get the safety and performance of a single player with index `i`, just do
```julia
(s, p) = f(prodFunc, i, xs, xp)
```
or, equivalently,
```julia
(s, p) = prodFunc(i, xs, xp)
```
where `xs` and `xp` are both scalar values; the outputs `s` and `p` will also be scalar.

### The `RiskFunc` type

The `RiskFunc.jl` file implements a `RiskFunc` type, which represents *σ<sub>i</sub>* in the model (the probability of a safe outcome given that player *i* is the contest winner). The default risk function is `WinnerOnlyRisk`, which defines:

$$\sigma_i(s) = \frac{s_i}{1+s_i}$$

That is, the probability of a safe outcome is determined entirely by the safety *s* of whoever wins the contest, with *s<sub>i</sub>* interpreted as the *odds* of a safe outcome, given that player *i* wins the contest.


If you don't like this assumption, you can change how the risk function is defined. Some options are pre-defined in `RiskFunc.jl`, with another reasonable option being `MultiplicativeRisk`, which defines, for some vector of weights *w*:

$$\sigma_i(s) = \left[ \prod_j \left(\frac{s_j}{1+s_j}\right)^{w_j} \right]^{n/\sum_j w_j}$$

That is, the probability of a safe outcome is, regardless of who wins the contest, *n* times the weighted geometric average of each player's individual probability *s<sub>i</sub> / (1+s<sub>i</sub>)*. If *w<sub>i</sub>* is the same for all players, then this is just

$$\sigma_i(s) = \prod_j \left(\frac{s_j}{1+s_j}\right),$$

which we can interpret as the case where each player has an *independent* probability *1 / (1+s<sub>i</sub>)* of causing a disaster, and *&sigma;<sub>i</sub>* is the probability that no player causes a disaster.

There is also an `AdditiveRisk` option implemented, which defines:

$$\sigma_i(s) = \frac{\sum_j w_j \left(\frac{s_j}{1+s_j}\right)}{\sum_j w_j}$$

### The `CSF` type

The `CSF.jl` file implements a `CSF` (contest success function) type, which represents *q<sub>i</sub>* in the model (the probability that player *i* wins the contest). The only version currently implemented is `BasicCSF`, which defines:

$$q_i(p) = \frac{p_i}{\sum_j p_j}$$

### The `PayoffFunc` type

The `PayoffFunc.jl` files implements a `PayoffFunc` type, which represents *&rho;<sub>ij</sub>* in the model (the payoff that player *i* gets if player *j* wins the contest, and the outcome is safe). The only version currently implemented is `LinearPayoff`, which defines

$$\rho_{ij}(p) = \begin{cases}
    a_w + b_w p, & i = j \\
    a_l + b_l p, & i \neq j
\end{cases}$$

for constants *a<sub>w</sub>*, *b<sub>w</sub>*, *a<sub>l</sub>*, and *b<sub>l</sub>*.

As a default, it is assumed that *a<sub>w</sub> = 1* and *b<sub>w</sub> = a<sub>l</sub> = b<sub>l</sub> = 0*, so a player gets a payoff of 1 if they win, and a payoff of zero otherwise.

### The `CostFunc` type

The `CostFunc.jl` file implements a `CostFunc` type, which controls the cost *c<sub>i</sub>* that players pay to use inputs *X<sub>s</sub>*, *X<sub>p</sub>*. The default `CostFunc` is `FixedUnitCost`, which represents the case where players pay a constant marginal cost *r* for *X<sub>s</sub>* and *X<sub>p</sub>*, described by
$$c_i(X_s, X_p) = r_i(X_{s,i} + X_{p,i}).$$

Notice that `FixedUnitCost` uses the same unit cost for both *X<sub>s</sub>* and *X<sub>p</sub>*, although we can provide a different *r<sub>i</sub>* for each player. If we want to allow *X<sub>s</sub>* and *X<sub>p</sub>* to have different unit costs, we can use `FixedUnitCost2`, which implements
$$c_i(X_s, X_p) = r_{s,i} X_{s,i} + r_{p,i} X_{p,i}.$$

There are a few other options for `CostFunc`s defined, which you can play with if you want to go poking around in the `CostFunc.jl` file. And, of course, you can always define your own.

### The `Problem` type

The `Problem.jl` file implements a `Problem` type that represents the payoff function

$$u_i := \sum_{j=1}^n \sigma_j(s) q_j(p) \rho_{ij}(p) - \left( 1 - \sum_{j=1}^n \sigma_j(s) q_j(p) \right) d_i - c_i(X_s, X_p)$$

You can construct a `Problem` like this:
```julia
problem = Problem(
    n = 2,  # default is 2
    d = 1.,
    r = 0.05,
    prodFunc = yourProdFunc,
    riskFunc = yourRiskFunc,  # default is WinnerOnlyRisk()
    csf = yourCSF,  # default is BasicCSF()
    payoffFunc = yourPayoffFunc,  # default is LinearPayoff(1, 0, 0, 0)
)
```
Note that the lengths of `d` and `r` must match and be equal to `n` and `prodFunc.n`. Again, you can omit arguments to use default values or provide vectors instead of scalars if you want different values for each player.

Instead of providing `r`, you can provide a `CostFunc` with the keyword `costFunc`, e.g., `costFunc = FixedUnitCost(2, [0.1, 0.1])`. If you just provide `r`, it will be interpreted as `costFunc = FixedUnitCost(n, r)`.

[Tip: to get help for a function in Julia, just type `?` then the function name in the Julia REPL. For, example, `?Problem` will give you something like the following:

```
[...]

Handy multi-purpose constructor for Problem type

  provide all params as keyword arguments

  if n is not provided, it defaults to n = 2

  d can be provided as a scalar or a vector of length n, if not provided, defaults to 0

  to specify the costFunc to use, provide one of the following:

    •  costFunc, which isa pre-constructed CostFunc

    •  rs and rp as scalars or vectors of length n; this will be interpreted as
       FixedUnitCost2(rs, rp)

    •  r as a scalar or vector of length n; this will be interpreted as
       FixedUnitCost([n,] r)

  If you provide more than one of the above, the first one in the list above will be
  used. If none of the above are provided, will default to FixedUnitCost(n, 0.1)

  to specify the prodFunc to use, provide one of the following:

    •  prodFunc, which isa pre-constructed ProdFunc

    •  arguments for ProdFunc constructor, e.g., one or more of (A, α, B, β, θ)

  If you provide more than one of the above, the first one in the list above will be
  used. If none of the above are provided, will default to ProdFunc()

  riskFunc, csf, and payoffFunc can be provided as pre-constructed objects; otherwise
  defaults riskFunc = WinnerOnlyRisk(), csf = BasicCSF(), payoffFunc = LinearPayoff(n)
  will be used
```
This gives you some help on how to create a new `Problem`.]

To calculate the payoffs for all the players, you can do
```julia
payoffs = get_payoffs(problem, Xs, Xp)
```
or
```julia
payoffs = problem(Xs, Xp)
```

and for just player `i`,
```julia
payoff_i = get_payoff(problem, i, Xs, Xp)
```
or
```julia
payoff_i = problem(i, Xs, Xp)
```

(Note that in the above, `Xs` and `Xp` are vectors of length `problem.n`.)

## Solvers

The `solve.jl` file contains several methods for finding Nash equilibria for a given problem. You can call all of these using the `solve` function, which takes a problem and optional keyword arguments and returns a `SolverResult` (which is basically just a container for the equilibrium values of safety, performance, and payoffs, plus an indicator for whether the solver converged successfully).

By default, `solve(problem)` will find a pure strategy solution for `problem` using a method of iterating on players' best responses: it starts with an arbitrary choice of *X<sub>s</sub>* and *X<sub>p</sub>* and at each iteration figures out the choice of strategy for each player that will maximize their payoff given the others' strategies. When the best-response strategies stop changing significantly at each iteration, we've reached a Nash equilibrium. You can also explicitly specify that you want to use this method by calling:
```julia
solve(problem, method = :iters)
```

You can also supply keyword arguments to modify some details of the solver's behavior. For example:
```julia
solve(
    problem,
    method = :iters,
    tol = 1e-8,  # stop if max relative diff. between iterations is < 1e-8
    max_iters = 1_000,  # go up to 1k iterations
    verbose = true  # print updates while solving
)
```
To see what options are available, take a look at the fields in the `SolverOptions` struct in `solve.jl`. (You should be able to access these using the command `fieldnames(AIIncentives.SolverOptions)`.)

The other methods you can use are the following:

* `method = :scatter` runs the iterating method with multiple, randomly-selected, starting points. The returned `SolverResult` will contain the solutions from each of those starting points. (Ideally, the solutions should all be the same.) This is typically just helpful for figuring out if solutions are sensitive to the starting point.

* `method = :mixed` runs a variation of the iterating method that attempts to maximize the best responses over a *history* of strategies. If the size of that history is large enough and the solver is run for enough iterations, the result should be a sample from a mixed strategy Nash equilibrium. You can control the history size by setting the `n_points` keyword argument. For example,
    ```julia
    solve(problem, method = :mixed, n_points = 100)
    ```
    will return a sample of size 100 from a (proposed) mixed-strategy equilibrium for `problem`. Note that this method can be very slow, but this is the only method that can give you mixed-strategy solutions.

* `method = :hybrid` starts by attempting to find a solution with the basic iterating method. If that fails, it will call `solve` again using `method = :mixed`; however, it will still only accept a pure-strategy solution -- this is helpful for finding some equilibria that the basic iterating method can't find. If this method fails, then it's likely that the only solutions are mixed-strategy equilibria.

## Scenarios

The `scenarios.jl` file includes some helpful tools for looking at cases where you vary one or two variables of a problem while holding the others fixed.

The main type defined here is `Scenario`, which is essentially an object that generates a family of related `Problem`s. You can create a scenario like
```julia
scenario = Scenario(
    n = 2,
    A = 10.,
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = 0.25, 
    d = 1.,
    r = range(0.01, 0.1, length = 20),
    varying = :r
)
```
which defines a 2-player scenario where `r` varies over 20 values between 0.01 and 0.1. Notice that we construct the scenario with all the variables we would normally provide to create a `ProdFunc` and `Problem`. Most of the arguments are single values, but the parameter we want to vary is not; that parameter must be an array with a column for each player and a row for each value we want to use. (You can also provide a single vector, in which case it will be assumed that you want to use the same values for all players.) We also need to specify which parameter we're varying with the `varying` keyword argument (if not included, the default is `:r`). 

If we want, we can include a second varying parameter, like so:
```julia
scenario = Scenario(
    n = 2,
    A = 10.,
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = range(0., 1., length = 4),
    d = 1.,
    r = range(0.01, 0.1, length = 20),
    varying = :r,
    varying2 = :θ
)
```
In this example, we look at the problem with every combination of the provided varying and secondary varying parameters. The only difference between the two is that when we plot the results, the varying parameter will vary along the x-axis, while the secondary varying parameter will vary in different series.

In the scenarios we've seen so far, we've provided only single values to the non-varying parameters. If we want the players to have different parameters, we need to provide a vector (with length equal to `n`) instead. For example,
```julia
scenario = Scenario(
    n = 2,
    A = [10., 20.],
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = 0.25, 
    d = 1.,
    r = range(0.01, 0.1, length = 20),
    varying = :r
)
```
gives us a scenario where player 1 has A = 10, and player 2 has A = 20.

You can also supply `riskFunc`, `csf`, and `payoffFunc` arguments to the scenario constructor if you want to use something other than the defaults, e.g.:
```julia
scenario = Scenario(
    n = 2,
    r = range(0.01, 0.1, length = 20),
    varying = :r,
    riskFunc = WinnerOnlyRisk(),
    payoffFunc = LinearPayoff(1., 0.1, 0., 0.)
)
```

To find the equilibrium solutions for a scenario, use the `solve` function:
```julia
results = solve(scenario)
```
This will return a `ScenarioResult` object, which just packages together the given scenario and a vector or array of `SolverResult` objects.

You can specify the method you want to use to solve with the `method` keyword argument:
```julia
results = solve(scenario, method = :mixed)
```
The default method is `:iters`.

Like with the `solve` function for problems, you can also supply extra options as keyword arguments:
```julia
results = solve(
    scenario,
    method = :mixed, 
    n_points = 100,
    verbose = true
)
```

You can plot the results as
```julia
plot(result)
```
where `result` is a `ScenarioResult`. You can provide arguments to customize the plot:
```julia
plot(
    result,
    plotsize = (900, 800),
    title = "Mixed strategy solutions",
    logscale = true,
    take_avg = true
)
```

The `plot` method will return a plot object that you can edit with the `Plots.jl` package. This plot will have 6 subplots, but if you only want one of those subplots, you can use `get_plots` instead of `plot` to get a list of those subplots. For example, if we want just the plot of the probability-weighted *&sigma;*, we can run
```julia
get_plots(result)[5]
```
and if we want to change the formatting, we can do that with the typical `Plots.jl` API. For example,
```julia
plot!(plot_title = "Some fantastic plots", xlabel = "a new label for the x axis")
```
would change the title and x-axis label for the most recently created plot.

## Heterogeneous beliefs

By default, actors in problems/scenarios are assumed to have perfect information (correct beliefs) about all parameters (their own and others'). The package also supports some special problems and scenarios that describe situations where players can have different beliefs about the model parameters. Currently, we can assume only that players vary in their beliefs about the model state but that their higher-order beliefs are perfect (i.e., they may disagree about the true model parameters but are perfectly informed about their competitors' beliefs).

### `ProblemWithBeliefs`

The `ProblemWithBeliefs` type is essentially a container that holds

1. a `Problem` called `baseProblem` which describes the *true* parameter values
2. a vector of `Problem`s called `beliefs`, which describe the `Problem`s that each player thinks they are in

For example, if we let the first element of `beliefs` be equal to `baseProblem`, then we're assuming that player 1 has correct beliefs; if we let the second element of `beliefs` be the same as `baseProblem`, but with `A` higher, then we're assuming that player 2 thinks that safety productivity is higher than it actually is.

Here's how we could initialize the above example:
```julia
problem = ProblemWithBeliefs(
    Problem(A = 10),  # this is baseProblem
    beliefs = [
        Problem(A = 10),  # same as baseProblem in this example
        Problem(A = 20)  # player 2 has different belief about A
    ]
)
```

We can use `ProblemWithBeliefs` in essentially the same way we might use a `Problem`; i.e., we can plug it into `solve`, get payoffs for a particular strategy, and so on.

### `ScenarioWithBeliefs`

What if we want to create a `Scenario` with heterogenous beliefs? Here, we can use the handy `ScenarioWithBeliefs`, which works analogously to `ProblemWithBeliefs` (as described above). To create a `ScenarioWithBelefs`, we supply a `baseScenario`, which represents the true parameter values for the scenario; then we supply `beliefs`, which here is a vector of `Dict`s that indicate how each player's beliefs differ from `baseScenario`.

As an example, if we wanted a scenario with `A = 10`, player 1's beliefs match the true parameters (with `A = 10`), and player 2's believes that `A = 20`, we could let
```julia
beliefs = [
    Dict(),  # supply an empty Dict to denote no difference from baseScenario
    Dict(:A => 20)
]
```

To initialize the complete scenario:
```julia
scenario = ScenarioWithBeliefs(
    # supply baseScenario first:
    Scenario(
        A = 10,
        # ... other scenario params
    ),
    beliefs = [
        Dict(),
        Dict(:A => 20)
    ]
)
```

We can use `ScenarioWithBeliefs` just like we would a normal `Scenario`; i.e., we can plug it into `solve`, plot the result, and so on.
