# Exact Riemann solver: Toro's Table 9.2 wave patterns, far-field and star
# states, Rankine-Hugoniot consistency of the shock speeds, and fan continuity.
using Test

include(joinpath(@__DIR__, "..", "experiments", "riemann", "exact_riemann_solver.jl"))

@testset "exact Riemann solver, test $(t.id)" for t in TORO_TESTS
    kappa = toro_kappa()
    sol = RiemannSolution(t.AL, t.uL, t.AR, t.uR, 25.0, kappa)
    @test wave_pattern(sol) == t.pattern

    @test all(sample(sol, -1.0e6) .≈ (t.AL, t.uL))
    @test all(sample(sol, 1.0e6) .≈ (t.AR, t.uR))

    eps = 1.0e-7 * max(1.0, abs(sol.u_star))
    for xi in (sol.SL_tail + eps, sol.SR_tail - eps)
        A, u = sample(sol, xi)
        @test A ≈ sol.A_star rtol = 1e-5
        @test u ≈ sol.u_star rtol = 1e-5 atol = 1e-6
    end

    if sol.left_wave === :shock
        @test (sol.uL - sol.SL_head) * sol.AL ≈ sol.A_star * (sol.u_star - sol.SL_head) rtol = 1e-8
        @test sol.u_star - sol.uL ≈ -first(f_wave(sol.A_star, sol.AL, kappa)) rtol = 1e-5
    else
        A, u = sample(sol, sol.SL_head + 1e-9)
        @test A ≈ sol.AL rtol = 1e-6
        @test u ≈ sol.uL rtol = 1e-6 atol = 1e-6
    end
    if sol.right_wave === :shock
        @test (sol.SR_head - sol.uR) * sol.AR ≈ sol.A_star * (sol.SR_head - sol.u_star) rtol = 1e-8
        @test sol.u_star - sol.uR ≈ first(f_wave(sol.A_star, sol.AR, kappa)) rtol = 1e-5
    else
        A, u = sample(sol, sol.SR_head - 1e-9)
        @test A ≈ sol.AR rtol = 1e-6
        @test u ≈ sol.uR rtol = 1e-6 atol = 1e-6
    end
end
