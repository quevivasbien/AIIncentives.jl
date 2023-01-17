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

Next, you'll need to install the project code. The easiest way to do this is to open a new Julia REPL session, then run
```
] add https://github.com/quevivasbien/AIIncentives.jl
```
which will automatically download and install the project as a package on your computer. (You'll only need to do this last step once, but after that you can run `] update AIIncentives` every now and then to make sure you're up to date with the latest available version.)

You should now be able import the package:
```julia
using AIIncentives
```

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
[
    
A couple of tangential tips here:

- To use math-related characters in most Julia code, you can typically just type the Latex code then press Tab. For example, `\alpha` + `[Tab]` becomes `α`. I've also allowed the spelled-out versions of Greek letters to work in most places in this code; for example, you could use `alpha`, `beta`, and `theta` instead of `α`, `β`, and `θ` in the example above.

- You can speed up a lot of solve calls by enabling multithreading. To do that, follow the instructions [here](https://docs.julialang.org/en/v1/manual/multi-threading/).

]

This will find Nash equilibria over the given parameterization and range of values for `r`. To generate a plot of the result, you can execute
```julia
plot(solution)
```

The first time you run something in the Julia session, it will take a while, since Julia compiles your code the first time you run it in a new setting. It should run a lot faster after that.

For more detailed documention, see [this document](./docs/docs.md).