# ai-julia

## What is this?

The code in [this repository](https://github.com/quevivasbien/ai-julia) is meant to find Nash equilibria for the following model:

We assume that *n* players produce safety, *s*, and performance, *p*, as

<div style="text-align: center">

![s_i = A_i X_{s,i}^{\alpha_i} p_i^{-\theta_i}, \quad p_i = B_i X_{p,i}^{\beta_i}](https://latex.codecogs.com/svg.image?s_i&space;=&space;A_i&space;X_{s,i}^{\alpha_i}&space;p_i^{-\theta_i},&space;\quad&space;p_i&space;=&space;B_i&space;X_{p,i}^{\beta_i} "formula for p and s")

</div>

for *i = 1, ..., n*. The *X* are inputs chosen by the players, and all other variables are fixed parameters.

In a Nash equilibrium, each player *i* chooses *X<sub>s,i</sub>* and *X<sub>p,i</sub>* to maximize the payoff

<div style="text-align: center">

![\pi_i := \left( \prod_{j=1}^n \frac{s_j}{1+s_j} \right) \rho_i(p) - \left( 1 - \prod_{j=1}^n \frac{s_j}{1+s_j} \right) d_i - r_i(X_{i,s} + X_{i,p})](https://latex.codecogs.com/svg.image?\pi_i&space;:=&space;\left(&space;\prod_{j=1}^n&space;\frac{s_j}{1&plus;s_j}&space;\right)&space;\rho_i(p)&space;-&space;\left(&space;1&space;-&space;\prod_{j=1}^n&space;\frac{s_j}{1&plus;s_j}&space;\right)&space;d_i&space;-&space;r_i(X_{i,s}&space;&plus;&space;X_{i,p}) "formula for payoff")

</div>

subject to the other players' choices of *X<sub>s</sub>* and *X<sub>p</sub>*. Here *ρ<sub>i</sub>(p)* is a contest success function (the expected payoff for player *i* given a safe outcome and a vector of performances *p*), and *d<sub>i</sub>* is the damage incurred by player *i* in the event of an unsafe outcome.

## Getting started

If you don't have Julia, download it from https://julialang.org/downloads/ and install. You'll then need to install the packages that this project uses, which with Julia is luckily quite simple: open a console window, type `julia` to enter an interactive Julia session, then run
```julia
using Pkg
Pkg.add(["Optim", "NLSolve", "Plots"])
```
You should then be able to use the code. From an interactive Julia session in the same directory as the code, run `include("scenarios.jl")` to import the project code.

You can then create and solve a scenario like
```julia
scenario = Scenario(
    2,  # n_players
    [10., 10.],  # A
    [0.5, 0.5],  # α
    [10., 10.],  # B
    [0.5, 0.5],  # β
    [0.25, 0.25],  # θ
    [1., 1.],  # d
    linspace(0.01, 0.1, 20),  # r
    varying_param = :r
)

solution = solve(scenario)
```

This will find Nash equilibria over the given parameterization and range of values for `r`. To generate a plot of the result, you can execute
```julia
plot_result(solution)
```

The first time you run something in the Julia session, it will take a while, since Julia compiles your code the first time you run it in a new setting. It should run a lot faster after that. Because of this, I recommend working within a Julia REPL session or a Jupyter notebook.

## Base types

### The `ProdFunc` type

The `ProdFunc.jl` file implements a `ProdFunc` (production function) type that contains the variables for the function
<div style="text-align: center">

![f(X_s, X_p) = (AX_s^\alpha (BX_p^\beta)^{-\theta},\ BX_p^\beta)](https://latex.codecogs.com/svg.image?f(X_s,&space;X_p)&space;=&space;(AX_s^\alpha&space;(BX_p^\beta)^{-\theta},\&space;BX_p^\beta) "formula for production function")

</div>

describing how the inputs *X<sub>s</sub>* and *X<sub>p</sub>* produce safety and performance for all players. You can create an instance of the `ProdFunc` type like
```julia
prodFunc = ProdFunc(
    2,  # n_players
    [10., 10.],  # A
    [0.5, 0.5],  # α
    [10., 10.],  # B
    [0.5, 0.5],  # β
    [0.25, 0.25]  # θ
)
```
You don't need to include the `n_players` parameter, but either way the other parameters need to all be vectors of equal length (equal to `n_players`). You can also provide all parameters as keyword arguments, in which case any excluded parameters will be set to default values; e.g., the following is equivalent to the above:
```julia
prodFunc = ProdFunc(
    A = [10., 10.],
    α = [0.5, 0.5],
    B = [10., 10.],
    β = [0.5, 0.5],
    θ = [0.25, 0.25]
)
```
(Side note: to use math-related characters in most Julia code, you can typically just type the Latex code then press Tab. For example, `\alpha` + `[Tab]` becomes `α`.)

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

### The `CSF` type

The `ProdFunc.jl` file also implements a `CSF` (contest success function) type, which represents the *ρ* function in the model. The constructor takes four parameters, like in the following example:

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

You can also get derivatives with `all_reward_derivs` and `reward_deriv`.

### The `Problem` type

The `Problem.jl` file implements a `Problem` type that represents the payoff function

<div style="text-align: center">

![\pi_i := \left( \prod_{j=1}^n \frac{s_j}{1+s_j} \right) \rho_i(p) - \left( 1 - \prod_{j=1}^n \frac{s_j}{1+s_j} \right) d_i - r_i(X_{i,s} + X_{i,p})](https://latex.codecogs.com/svg.image?\pi_i&space;:=&space;\left(&space;\prod_{j=1}^n&space;\frac{s_j}{1&plus;s_j}&space;\right)&space;\rho_i(p)&space;-&space;\left(&space;1&space;-&space;\prod_{j=1}^n&space;\frac{s_j}{1&plus;s_j}&space;\right)&space;d_i&space;-&space;r_i(X_{i,s}&space;&plus;&space;X_{i,p}) "formula for payoff").

</div>

You can construct a `Problem` like this:
```julia
problem = Problem(
    2,  # n (number of players, optional)
    [1., 1.],  # d (disaster cost)
    [0.05, 0.05],  # r (rental cost)
    prodFunc,  # a ProdFunc
    csf  # a CSF
)
```
Note that the lengths of `d` and `r` must match and be equal to `n` and `prodFunc.n_players`.

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

The `solve.jl` file contains several methods for finding Nash equilibria for a given problem. Your default choice should probably be `solve_iters`, which finds pure strategy equilibria by iterating on each player's best responses: it starts with an arbitrary choice of *X<sub>s</sub>* and *X<sub>p</sub>* and at each iteration figures out the choice of strategy for each player that will maximize their payoff given the others' strategies. When the best-response strategies stop changing significantly at each iteration, we've reached a Nash equilibrium. You can use this method just as
```julia
solve_iters(problem)
```
where `problem` is a `Problem`, which will return a `SolverResult` object (basically just a container for the equilibrium values of safety, performance, and payoffs; the `SolverResult` also has a `success` field which indicates exactly what it says -- true iff the solver was successful).

You can also supply a `SolverOptions` object as a second argument, to modify some details of how the solver works. For example:
```julia
options = SolverOptions(
    tol = 1e-8,  # stop if max relative diff. between iterations is < 1e-8
    max_iters = 1_000,  # go up to 1k iterations
    verbose = true  # print updates while solving
)

solve_iters(problem, options)
```

The other solvers available are `solve_roots`, `solve_hybrid`, `solve_scatter`, and `solve_mixed`.

`solve_roots` attempts to find pure strategy equilibria by solving the first order conditions. This is typically quite fast but fails more often than `solve_iters`, returning bogus results.

`solve_hybrid` starts by finding candidate solutions with `solve_roots`, then plugs them into `solve_iters` to check them. This is more reliable than `solve_roots`, but you should typically just use `solve_iters`.

`solve_scatter` runs `solve_iters` with multiple, randomly-selected, starting points. The returned `SolverResult` will contain the solutions from each of those starting points. (Ideally, the solutions should all be the same.)

`solve_hybrid` runs a variation of `solve_iters` that attempts to maximize the best responses over a *history* of strategies. If the size of that history is large enough and the solver is run for enough iterations, the result should be a sample from a mixed strategy Nash equilibrium. You can control the history size by setting `n_points` in a `SolverOptions` object you provide to the solver. For example,
```julia
options = SolverOptions(n_points = 100)
solve_mixed(problem, options)
```
will return a sample of size 100 from a (proposed) mixed-strategy equilibrium for `problem`.

## Scenarios

The `scenarios.jl` file includes some helpful tools for looking at cases where you vary one or two variables of a problem while holding the others fixed.

The main type defined here is `Scenario`. You can create a scenario like
```julia
scenario = Scenario(
    2,  # n_players
    [10., 10.],  # A
    [0.5, 0.5],  # α
    [10., 10.],  # B
    [0.5, 0.5],  # β
    [0.25, 0.25],  # θ
    [1., 1.],  # d
    range(0.01, 0.1, length = 20),  # r
    varying_param = :r
)
```
which defines a 2-player scenario where `r` varies over 20 values between 0.01 and 0.1. Notice that we construct the scenario with all the variables we would normally provide to create a `ProdFunc` and `Problem`. Most of the arguments are vectors of equal length (equal to `n_players`), but the parameter we want to vary is not; that parameter must be an array with a column for each player and a row for each value we want to use. (You can also provide a single vector, in which case it will be assumed that you want to use the same values for all players.) We also need to specify which parameter we're varying with the `varying_param` keyword argument (if not included, the default is `:r`). 

If we want, we can include a second varying parameter, like so:
```julia
scenario = Scenario(
    2,  # n_players
    [10., 10.],  # A
    [0.5, 0.5],  # α
    [10., 10.],  # B
    [0.5, 0.5],  # β
    range(0., 1., length = 4),  # θ
    [1., 1.],  # d
    range(0.01, 0.1, length = 20),  # r
    varying_param = :r,
    secondary_varying_param = :θ
)
```
In this example, we look at the problem with every combination of the provided varying and secondary varying parameters. The only difference between the two is that when we plot the results, the varying parameter will vary along the x-axis, while the secondary varying parameter will vary in different series.

To change the CSF used in a scenario, you can provide to the `Scenario` constructor keyword arguments for `w`, `l`, `a_w`, and `a_l`. (The defaults are 1 for `w` and 0 for the others.)

To find the equilibrium solutions for a scenario, use the `solve` function:
```julia
results = solve(scenario)
```
This will return a `ScenarioResult` object, which just packages together the given scenario and a vector or array of `SolverResult` objects.

You can specify the method you want to use to solve with the `method` keyword argument:
```julia
results = solve(scenario, method = :mixed)
```
The available options are `:iters`, `:roots`, `:hybrid`, `:scatter`, and `:mixed`. The default is `:iters`.

You can also provide a `SolverOptions` object with the `options` keyword:
```julia
results = solve(
    scenario,
    method = :mixed, 
    options = SolverOptions(n_points = 100, verbose = true)
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

## More about plotting

To-do
