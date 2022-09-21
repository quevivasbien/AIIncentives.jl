module AIIncentives

export
    solve,
    
    ProdFunc,
    MultiplicativeRisk,
    AdditiveRisk,
    WinnerOnlyRisk,
    LinearPayoff,
    PayoffOnDisaster,
    Problem,
    ProblemWithBeliefs,
    SolverOptions,
    SolverResult,
    Scenario,
    ScenarioResult,
    ScenarioWithBeliefs,

    get_plots,
    get_plots_for_result,  # deprecated
    plot,
    plot_result,  # deprecated
    plot_payoffs_with_xs,
    plot_payoffs_with_xp,
    plot_payoffs,
    plot_payoffs_near_solution,

    f,
    payoff,
    payoffs,
    reward,
    all_rewards


include("includes.jl")

end
