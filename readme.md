# AIIncentives.jl

## What is this?

The code in this repository is meant to find Nash equilibria for the following model:

We assume that *n* players produce safety, *s*, and performance, *p*, as

<div style="text-align: center">

![s_i = A_i X_{s,i}^{\alpha_i} p_i^{-\theta_i}, \quad p_i = B_i X_{p,i}^{\beta_i}](https://latex.codecogs.com/svg.image?s_i&space;=&space;A_i&space;X_{s,i}^{\alpha_i}&space;p_i^{-\theta_i},&space;\quad&space;p_i&space;=&space;B_i&space;X_{p,i}^{\beta_i} "formula for p and s")

</div>

for *i = 1, ..., n*. The *X* are inputs chosen by the players, and all other variables are fixed parameters.

In a Nash equilibrium, each player *i* chooses *X<sub>s,i</sub>* and *X<sub>p,i</sub>* to maximize the payoff

<div style="text-align: center">

![u_i := \sigma(s, p) \rho_i(p) - \left( 1 - \sigma(s, p) \right) d_i - r_i(X_{i,s} + X_{i,p})](https://latex.codecogs.com/svg.image?u_i&space;:=&space;\sigma(s,&space;p)&space;\rho_i(p)&space;-&space;\left(&space;1&space;-&space;\sigma(s,&space;p)&space;\right)&space;d_i&space;-&space;r_i(X_{i,s}&space;&plus;&space;X_{i,p}) "formula for payoff")

</div>

subject to the other players' choices of *X<sub>s</sub>* and *X<sub>p</sub>*. Here *σ* is a "risk function" that determines the probability of a safe outcome (no disaster), *ρ<sub>i</sub>* is a contest success function (the expected payoff for player *i* given a safe outcome and a vector of performances *p*), and *d<sub>i</sub>* is the damage incurred by player *i* in the event of a disaster.

## Getting started

If you don't have Julia, download it from https://julialang.org/downloads/ and install. You'll then need to install the packages that this project uses, which with Julia is luckily quite simple: open a console window, type `julia` to enter an interactive Julia session, then run
```
] add Optim, NLsolve, Plots
```
Those dependencies will now be installed, so you won't have to repeat that last step in the future.

At this point, the easiest way to load the project code is to open a new Julia session in the project directory -- from your computer's terminal, navigate to the project directory and run:
```bash
~/path/to/AIIncentives$ julia --project --threads=auto

julia> using AIIncentives
```
(The `--project` flag allows you to import the project with the `using` keyword, and the `--threads=auto` flag allows Julia to use all of your computer's CPU cores, which can speed some tasks up significantly.)

You will then be able to use the project code. If this is the first time you do this, there will be a bit of delay as it precompiles some of the code.

You can then create and solve a scenario like
```julia
scenario = Scenario(
    n_players = 2,
    A = 10.,
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = 0.25, 
    d = 1.,
    r = range(0.01, 0.1, length = 20),
    varying_param = :r
)

solution = solve(scenario)
```
(Side note: to use math-related characters in most Julia code, you can typically just type the Latex code then press Tab. For example, `\alpha` + `[Tab]` becomes `α`.)

This will find Nash equilibria over the given parameterization and range of values for `r`. To generate a plot of the result, you can execute
```julia
plot_result(solution)
```

The first time you run something in the Julia session, it will take a while, since Julia compiles your code the first time you run it in a new setting. It should run a lot faster after that. Because of this, I recommend working within a Julia REPL session or a Jupyter notebook.

If you just want to solve and plot scenarios where some parameter value changes, you can skip now to the section on the `Scenario` type. Otherwise, keep reading in a linear fashion.

## Base types

### The `ProdFunc` type

The `ProdFunc.jl` file implements a `ProdFunc` (production function) type that contains the variables for the function
<div style="text-align: center">

![f(X_s, X_p) = (AX_s^\alpha (BX_p^\beta)^{-\theta},\ BX_p^\beta)](https://latex.codecogs.com/svg.image?f(X_s,&space;X_p)&space;=&space;(AX_s^\alpha&space;(BX_p^\beta)^{-\theta},\&space;BX_p^\beta) "formula for production function")

</div>

describing how the inputs *X<sub>s</sub>* and *X<sub>p</sub>* produce safety and performance for all players. You can create an instance of the `ProdFunc` type like
```julia
prodFunc = ProdFunc(
    n_players = 2,
    A = 10.,
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = 0.25
)
```
If you want to supply different parameters for each player or just like typing more stuff out, you can also supply the parameters as vectors of equal length (equal to `n_players`). The following creates the same object as the example above:
```julia
prodFunc = ProdFunc(
    n_players = 2,
    A = [10., 10.],
    α = [0.5, 0.5],
    B = [10., 10.],
    β = [0.5, 0.5],
    θ = [0.25, 0.25]
)
```
If you omit a keyword argument, it will be set at a default value.

To determine the outputs for all players, you can do
```julia
(s, p) = f(prodFunc, Xs, Xp)
```
or, equivalently,
```julia
(s, p) = prodFunc(Xs, Xp)
```
where `Xs` and `Xp` are vectors of length equal to `prodFunc.n_players`. Here `s` and `p` will be vectors representing the safety and performance of each player. To get the safety and performance of a single player with index `i`, just do
```julia
(s, p) = f(prodFunc, i, xs, xp)
```
or, equivalently,
```julia
(s, p) = prodFunc(i, xs, xp)
```
where `xs` and `xp` are both scalar values; the outputs `s` and `p` will also be scalar.

You can also compute the Jacobian of `f` as `df`.

### The `RiskFunc` type

The `RiskFunc.jl` file implements a `RiskFunc` type, which represents the *σ* function in the model. The default risk function is the `MultiplicativeRiskFunc` defined as:

<div style="text-align: center">

![\sigma(s, p) = \prod_i \frac{s_i}{1+s_i}](https://latex.codecogs.com/svg.image?\sigma(s,&space;p)&space;=&space;\prod_i&space;\frac{s_i}{1&plus;s_i} "multiplicative risk function")

</div>

The interpretation is that each *s<sub>i</sub>* represents the *odds* that player *i* causes a disaster, with players' chances of causing a disaster independently distributed, and *σ* being the probability that *no player* causes a disaster.

If you don't like this assumption, you can change how the risk function is defined. Some options are pre-defined in `RiskFunc.jl`, with another reasonable option being the `WinnerOnlyRiskFunc`, defined as:

<div style="text-align: center">

![\sigma(s, p) = \sum_i \left(\frac{p_i}{\sum_j p_j}\right) \left(\frac{s_i}{1+s_i}\right)](https://latex.codecogs.com/svg.image?\sigma(s,&space;p)&space;=&space;\sum_i&space;\left(\frac{p_i}{\sum_j&space;p_j}\right)&space;\left(\frac{s_i}{1&plus;s_i}\right) "winner-only risk function")

</div>

This represents the assumption that only the winning player can cause a disaster (where one's probability of winning is one's performance divided by the sum of everyone's performance).

### The `CSF` type

The `CSF.jl` file implements a `CSF` (contest success function) type, which represents the *ρ* function in the model. The constructor takes four parameters, like in the following example:

```julia
csf = CSF(
    1.,  # w
    0.,  # l
    0.,  # a_w
    0.  # a_l
)
```

You can also provide all parameters as keyword arguments, in which case any excluded parameters will be set to default values. That is, the following is equivalent to the above:
```julia
csf = CSF(w = 1., l = 0., a_w = 0., a_l = 0.)
```

The parameters correspond to the following definition of *ρ*:

<div style="text-align: center">

![\rho_i = (w + a_w) \cdot \frac{p_i}{\sum_j p_j} + (l + a_l) \left( 1 - \frac{p_i}{\sum_j p_j} \right)](https://latex.codecogs.com/svg.image?\rho_i&space;=&space;(w&space;&plus;&space;a_w)&space;\cdot&space;\frac{p_i}{\sum_j&space;p_j}&space;&plus;&space;(l&space;&plus;&space;a_l)&space;\left(&space;1&space;-&space;\frac{p_i}{\sum_j&space;p_j}&space;\right) "formula for CSF")

</div>

To get the rewards for all players given a vector of performance `p`, you can do
```julia
rewards = all_rewards(csf, p)
```
or just
```julia
rewards = csf(p)
```
and to get the reward for just a single player, indexed by `i`,
```julia
reward_for_player_i = reward(csf, i, p)
```
or just
```julia
reward_for_player_i = csf(i, p)
```

### The `Problem` type

The `Problem.jl` file implements a `Problem` type that represents the payoff function

<div style="text-align: center">

![u_i := \sigma(s, p) \rho_i(p) - \left( 1 - \sigma(s, p) \right) d_i - r_i(X_{i,s} + X_{i,p})](https://latex.codecogs.com/svg.image?u_i&space;:=&space;\sigma(s,&space;p)&space;\rho_i(p)&space;-&space;\left(&space;1&space;-&space;\sigma(s,&space;p)&space;\right)&space;d_i&space;-&space;r_i(X_{i,s}&space;&plus;&space;X_{i,p}) "formula for payoff").

</div>

You can construct a `Problem` like this:
```julia
problem = Problem(
    n_players = 2,  # default is 2
    d = 1.,
    r = 0.05,
    prodFunc = yourProdFunc,
    riskFunc = yourRiskFunc,  # default is MultiplicativeRiskFunc(n_players)
    csf = yourCSF  # default is CSF(1, 0, 0, 0)
)
```
Note that the lengths of `d` and `r` must match and be equal to `n` and `prodFunc.n_players`. Again, you can omit arguments to use default values or provide vectors instead of scalars if you want different values for each player.

To calculate the payoffs for all the players, you can do
```julia
payoffs = all_payoffs(problem, Xs, Xp)
```
and for just player `i`:
```julia
payoff_for_player_i = payoff(problem, i, Xs, Xp)
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

The main type defined here is `Scenario`. You can create a scenario like
```julia
scenario = Scenario(
    n_players = 2,
    A = 10.,
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = 0.25, 
    d = 1.,
    r = range(0.01, 0.1, length = 20),
    varying_param = :r
)
```
which defines a 2-player scenario where `r` varies over 20 values between 0.01 and 0.1. Notice that we construct the scenario with all the variables we would normally provide to create a `ProdFunc` and `Problem`. Most of the arguments are single values, but the parameter we want to vary is not; that parameter must be an array with a column for each player and a row for each value we want to use. (You can also provide a single vector, in which case it will be assumed that you want to use the same values for all players.) We also need to specify which parameter we're varying with the `varying_param` keyword argument (if not included, the default is `:r`). 

If we want, we can include a second varying parameter, like so:
```julia
scenario = Scenario(
    n_players = 2,
    A = 10.,
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = range(0., 1., length = 4),
    d = 1.,
    r = range(0.01, 0.1, length = 20),
    varying_param = :r,
    secondary_varying_param = :θ
)
```
In this example, we look at the problem with every combination of the provided varying and secondary varying parameters. The only difference between the two is that when we plot the results, the varying parameter will vary along the x-axis, while the secondary varying parameter will vary in different series.

In the scenarios we've seen so far, we've provided only single values to the non-varying parameters. If we want the players to have different parameters, we need to provide a vector (with length equal to `n_players`) instead. For example,
```julia
scenario = Scenario(
    n_players = 2,
    A = [10., 20.],
    α = 0.5,
    B = 10.,
    β = 0.5,
    θ = 0.25, 
    d = 1.,
    r = range(0.01, 0.1, length = 20),
    varying_param = :r
)
```
gives us a scenario where player 1 has A = 10, and player 2 has A = 20.

You can also supply `riskFunc` and `csf` arguments to the scenario constructor if you want to use something other than the defaults, e.g.:
```julia
scenario = Scenario(
    n_players = 2,
    r = range(0.01, 0.1, length = 20),
    varying_param = :r,
    riskFunc = WinnerOnlyRiskFunc(),
    csf = CSF(1, 0, 0.01, 0.01)
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
plot_result(result)
```
where `result` is a `ScenarioResult`. You can provide arguments to customize the plot:
```julia
plot_result(
    result,
    plotsize = (900, 800),
    title = "Mixed strategy solutions",
    logscale = true,
    take_avg = true
)
```

The `plot_result` method will return a plot object that you can edit with the `Plots.jl` package. This plot will have 6 subplots, but if you only want one of those subplots, you can use `get_plots_for_result` instead of `plot_result` to get a list of those subplots.
