module AIIncentives

export
    solve,
    
    ProdFunc,

    RiskFunc,
    MultiplicativeRisk,
    AdditiveRisk,
    WinnerOnlyRisk,

    PayoffFunc,
    LinearPayoff,
    PayoffOnDisaster,

    CostFunc,
    FixedUnitCost,
    FixedUnitCost2,
    LinearCost,
    CertificationCost,
    RelativeSafetyCertificationCost,
    PenaltyCost,

    AbstractProblem,
    Problem,
    ProblemWithBeliefs,

    SolverOptions,
    SolverResult,

    AbstractScenario,
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

    σ,  # σ(::RiskFunc, ...)
    cost,  # cost(::CostFunc, ...)
    f,  # f(::ProdFunc, ...)
    payoff,  # payoff(::AbstractProblem, ...)
    payoffs,  # payoffs(::AbstractProblem, ...)
    is_symmetric


include("includes.jl")

end
