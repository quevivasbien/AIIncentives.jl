The code in this repository is meant to find Nash equilibria for the following model:

We assume that *n* players produce safety, *s*, and performance, *p*, as

<div style="text-align: center">

![s_i = A_i X_{s,i}^{\alpha_i} p_i^{-\theta_i}, \quad p_i = B_i X_{p,i}^{\beta_i}](https://latex.codecogs.com/svg.image?s_i&space;=&space;A_i&space;X_{s,i}^{\alpha_i}&space;p_i^{-\theta_i},&space;\quad&space;p_i&space;=&space;B_i&space;X_{p,i}^{\beta_i} "formula for p and s").

</div>

for *i = 1, ..., n*. The *K* are inputs chosen by the players, and all other variables are fixed parameters.

In a Nash equilibrium, each player *i* chooses *X<sub>s,i</sub>* and *X<sub>p,i</sub>* to maximize the payoff

<div style="text-align: center">

![\pi_i := \left( \prod_{j=1}^n \frac{s_j}{1+s_j} \right) \rho_i(p) - \left( 1 - \prod_{j=1}^n \frac{s_j}{1+s_j} \right) d_i - r_i(X_{i,s} + X_{i,p})](https://latex.codecogs.com/svg.image?\pi_i&space;:=&space;\left(&space;\prod_{j=1}^n&space;\frac{s_j}{1&plus;s_j}&space;\right)&space;\rho_i(p)&space;-&space;\left(&space;1&space;-&space;\prod_{j=1}^n&space;\frac{s_j}{1&plus;s_j}&space;\right)&space;d_i&space;-&space;r_i(X_{i,s}&space;&plus;&space;X_{i,p}) "formula for payoff"),

</div>

subject to the other players' choices of *X_s* and *X_p*. Here *ρ<sub>i</sub>(p)* is a contest success function (the expected payoff for player *i* given a safe outcome and a vector of performances *p*), and *d<sub>i</sub>* is the damage incurred by player *i* in the event of an unsafe outcome.
