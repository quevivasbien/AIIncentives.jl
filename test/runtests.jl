using AIIncentives
using Test

function test_solve()
    println("Running test on `solve.jl`...")
    prodFunc = ProdFunc([10., 10.], [0.5, 0.5], [10., 10.], [0.5, 0.5], [0.25, 0.25])
    riskFunc = MultiplicativeRisk(2)
    csf = CSF(1., 0., 0., 0.)
    problem = Problem(2, [1., 1.], [0.01, 0.01], riskFunc, prodFunc, csf)
    options = SolverOptions(verbose = true)
    println("With `solve_iters`:")
    @time solve_iters_sol = solve_iters(problem, options)
    print(solve_iters_sol)
    println("With `solve_roots`:")
    @time solve_roots_sol = solve_roots(problem, options)
    print(solve_roots_sol)
    println("With `solve_mixed`:")
    @time solve_mixed_sol = solve_mixed(problem, options)
    print(solve_mixed_sol)
    println("With `solve_hybrid`:")
    @time solve_hybrid_sol = solve_hybrid(problem, options)
    print(solve_hybrid_sol)
    return
end

function test_scenarios()
    println("Running test on `scenarios.jl` + `plotting.jl`...")

    scenario = Scenario(
        n_players = 2,
        α = [0.5, 0.75],
        θ = [0., 0.5],
        r = range(0.01, 0.1, length = Threads.nthreads()),
        varying2 = :θ
    )

    @time res = solve(scenario, verbose  = true)
    plot(res)
end

@testset "AIIncentives.jl" begin
    test_solve()
    test_scenarios()
end
