# Entropy-conservative / entropy-stable flux pair, over random and adversarial
# state pairs and two parameter sets (CGS inviscid; SI viscous with offsets).
using Test
using Trixi
using LinearAlgebra: dot
using Random: MersenneTwister
using BloodFlowDGSEM
using BloodFlowDGSEM: entropy_flux_potential
using Trixi: max_abs_speeds

const EQ_CGS = ArteryFlowEquations1D(1.06, 0.0, 1.0, pi * 0.7688^2, 0.078, 0.4e6, 0.5)
const EQ_SI = ArteryFlowEquations1D(1050.0, 4.0e-3, 1.1, pi * 0.003^2, 0.3e-3, 0.7e6, 0.5;
                                    p_ext = 1.0e4, p_d = 1.0e4)

# Fixed pairs (rest, subcritical, transcritical, strong jump, near vacuum,
# supercritical) and random pairs over four decades in A and Froude numbers in [-3, 3].
function state_pairs(equations; n_random = 200, seed = 20260805)
    A0 = equations.A0
    st(A, fr) = SVector(A, A * fr * wave_speed(A, equations))
    pairs = [(st(A0, 0.0), st(A0, 0.0)),
             (st(1.3 * A0, 0.1), st(1.3 * A0, 0.1)),
             (st(1.2 * A0, 0.05), st(0.9 * A0, -0.05)),
             (st(1.5 * A0, 1.19), st(0.7 * A0, 0.4)),
             (st(3.0 * A0, 0.8), st(0.3 * A0, -0.6)),
             (st(1e-6 * A0, 0.2), st(1e-5 * A0, -0.3)),
             (st(A0, 2.5), st(A0, 2.5)),
             (st(2.0 * A0, 3.0), st(1.0 * A0, 2.0))]
    rng = MersenneTwister(seed)
    for _ in 1:n_random
        A_l = A0 * 10.0^(4 * rand(rng) - 2)
        A_r = A0 * 10.0^(4 * rand(rng) - 2)
        push!(pairs, (st(A_l, 6 * rand(rng) - 3), st(A_r, 6 * rand(rng) - 3)))
    end
    return pairs
end

@testset "entropy-stable flux ($name)" for (name, equations) in ("CGS" => EQ_CGS,
                                                                  "SI" => EQ_SI)
    pairs = state_pairs(equations)
    jump_v(a, b) = cons2entropy(b, equations) - cons2entropy(a, equations)
    jump_psi(a, b) = entropy_flux_potential(b, equations) - entropy_flux_potential(a, equations)

    # cons2entropy = ∇η. Mixed tolerance: at A = A0 the closed form gives v1 = 0
    # exactly while AD of η differences two large terms.
    for (a, b) in pairs, u in (a, b)
        v_ad = Trixi.ForwardDiff.gradient(x -> entropy(x, equations), u)
        floor_v = SVector(equations.K_p, equations.rho * wave_speed(u[1], equations))
        @test all(abs.(cons2entropy(u, equations) .- v_ad) .<= 1e-10 .* (abs.(v_ad) .+ floor_v))
    end

    for (a, b) in pairs, u in (a, b)
        f = flux(u, 1, equations)
        @test flux_ec(u, u, 1, equations) ≈ f rtol = 1e-12
        @test flux_es(u, u, 1, equations) ≈ f rtol = 1e-12
        @test max_abs_speed(u, u, 1, equations) ≈ max_abs_speeds(u, equations)[1] rtol = 1e-14
    end

    for (a, b) in pairs
        @test flux_ec(a, b, 1, equations) ≈ flux_ec(b, a, 1, equations) rtol = 1e-12

        # Tadmor: ⟦v⟧ᵀ f_ec = ⟦ψ⟧
        dv = jump_v(a, b)
        f_ec = flux_ec(a, b, 1, equations)
        dpsi = jump_psi(a, b)
        @test abs(dot(dv, f_ec) - dpsi) / max(abs(dot(dv, f_ec)), abs(dpsi), 1e-30) < 1e-11

        # f_es - f_ec = -½ λ ⟦u⟧ exactly
        expected = -0.5 * max_abs_speed(a, b, 1, equations) * (b - a)
        d = flux_es(a, b, 1, equations) - f_ec
        @test maximum(abs, d - expected) <= 1e-12 * max(maximum(abs, expected), 1e-30)

        # Monotone entropy gradient, hence entropy stability.
        du = b - a
        @test dot(dv, du) / max(maximum(abs, dv) * maximum(abs, du), 1e-30) > -1e-12
        f_es = flux_es(a, b, 1, equations)
        production = dot(dv, f_es) - dpsi
        @test production / max(abs(dot(dv, f_es)), abs(dpsi), 1e-30) < 1e-11
        if a != b
            @test dot(dv, du) > 0
            @test production < 0
        end
    end
end
