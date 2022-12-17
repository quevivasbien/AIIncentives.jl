using AIIncentives
using Test

function test_solve()
    println("Running test on `solve.jl`...")
    problem = Problem()
    
    println("With `solve_iters`:")
    solve(problem, method = :iters, verbose = true)
    
    println("With `solve_mixed`:")
    solve(problem, method = :mixed, verbose = true)
    
    println("With `solve_hybrid`:")
    solve(problem, method = :hybrid, verbose = true)
    
    return
end

function test_problems()
    println("Running test on `Problems.jl` and associated components")

    println("Testing basic problem...")
    Problem() |> solve

    println("Testing problem with FixedUnitCost2...")
    Problem(rs = 1., rp = 2.) |> solve

    println("Testing problem with CertificationCost...")
    Problem(r0 = 2., r1 = 1., s_thresh = 5) |> solve

    println("Testing problem with MultiplicativeRisk...")
    Problem(n = 3, riskFunc = MultiplicativeRisk(3)) |> solve

    return
end

function test_scenarios()
    println("Running test on `scenarios.jl`")

    println("Testing basic scenario...")
    scenario = Scenario(
        n = 2,
        r = range(0.01, 0.1, length = Threads.nthreads()),
        varying = :r
    )
    solve(scenario, verbose = true)

    println("Testing scenario with secondary variation...")
    scenario2 = Scenario(
        n = 2,
        θ = [0., 0.5],
        r = range(0.01, 0.1, length = Threads.nthreads()),
        varying = :r,
        varying2 = :θ
    )
    solve(scenario2, verbose  = true)

    return
end

@testset "AIIncentives.jl" begin
    test_solve()
    test_problems()
    test_scenarios()
end
