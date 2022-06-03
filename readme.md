# ai-julia

## What is this?

The code in this repository is meant to find Nash equilibria for the following model:

We assume that *n* players produce safety, *s*, and performance, *p*, as

<div style="text-align: center">

![s_i = A_i X_{s,i}^{\alpha_i} p_i^{-\theta_i}, \quad p_i = B_i X_{p,i}^{\beta_i}](https://latex.codecogs.com/svg.image?s_i&space;=&space;A_i&space;X_{s,i}^{\alpha_i}&space;p_i^{-\theta_i},&space;\quad&space;p_i&space;=&space;B_i&space;X_{p,i}^{\beta_i} "formula for p and s")

</div>

for *i = 1, ..., n*. The *K* are inputs chosen by the players, and all other variables are fixed parameters.

In a Nash equilibrium, each player *i* chooses *X<sub>s,i</sub>* and *X<sub>p,i</sub>* to maximize the payoff

<div style="text-align: center">

![\pi_i := \left( \prod_{j=1}^n \frac{s_j}{1+s_j} \right) \rho_i(p) - \left( 1 - \prod_{j=1}^n \frac{s_j}{1+s_j} \right) d_i - r_i(X_{i,s} + X_{i,p})](https://latex.codecogs.com/svg.image?\pi_i&space;:=&space;\left(&space;\prod_{j=1}^n&space;\frac{s_j}{1&plus;s_j}&space;\right)&space;\rho_i(p)&space;-&space;\left(&space;1&space;-&space;\prod_{j=1}^n&space;\frac{s_j}{1&plus;s_j}&space;\right)&space;d_i&space;-&space;r_i(X_{i,s}&space;&plus;&space;X_{i,p}) "formula for payoff"),

</div>

subject to the other players' choices of *X_s* and *X_p*. Here *ρ<sub>i</sub>(p)* is a contest success function (the expected payoff for player *i* given a safe outcome and a vector of performances *p*), and *d<sub>i</sub>* is the damage incurred by player *i* in the event of an unsafe outcome.


## Instructions for use

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
    [0.5, 0.5],  # alpha
    [10., 10.],  # B
    [0.5, 0.5],  # beta
    [0.25, 0.25],  # theta
    [1., 1.],  # d
    linspace(0.01, 0.1, 20),  # r
    varying_param = :r
)

solve(scenario, method = :hybrid)
```

This will generate a plot of the proposed equilibria over the given parameterization and range of values for `r`. The first time you run something in the Julia session, it will take a while, since Julia compiles your code the first time you run it in a new setting. It should run a lot faster after that.
