# Windkessel outlet and the coupled 1D-0D ODE: junction algebra (SI and CGS), and
# on a one-element carotid (Boileau et al. Table III, CGS) the quiescent state,
# the steady RCR limit, purity of the right-hand side, the flux identity at every
# RK stage, volume conservation, temporal order 5, and step values with tstops.
using Test
using Trixi
using OrdinaryDiffEqLowStorageRK
using Printf: @printf
using SciMLBase: remake
using SimpleNonlinearSolve: NonlinearProblem, SimpleNewtonRaphson, solve
using BloodFlowDGSEM
using BloodFlowDGSEM: windkessel_residual, windkessel_nlfunc, windkessel_exterior,
                      inflow_residual, inflow_jacobian

residual_params(bc, u_inner, eq) = (u_inner, bc.R1, bc.p_C, 1, flux_es, eq)

function exact_root(bc, u_inner, eq)
    prob = NonlinearProblem(windkessel_nlfunc, log(u_inner[1]), residual_params(bc, u_inner, eq))
    return exp(solve(prob, SimpleNewtonRaphson(); abstol = 1e-15, reltol = 1e-15).u)
end

@testset "junction algebra" begin
    eq = ArteryFlowEquations1D(1060.0, 4e-3, 2.0, 0.22038e-4, 0.3e-3, 700e3, 0.5)
    bc = BoundaryConditionWindkessel3Element(2.4875e8, 1.8697e9, 1.7529e-10; p_C0 = 3.5e3)
    A_i, u_i = 1.07 * eq.A0, 0.11
    u_inner = SVector(A_i, A_i * u_i)

    # Unique root: the residual is strictly decreasing in log A_b.
    p = residual_params(bc, u_inner, eq)
    Ls = log.(range(0.5, 2.0, length = 50) .* eq.A0)
    @test all(diff([windkessel_residual(L, p) for L in Ls]) .< 0)
    for L in Ls[1:7:end]
        dF = Trixi.ForwardDiff.derivative(l -> windkessel_residual(l, p), L)
        h = 1e-6
        @test dF ≈ (windkessel_residual(L + h, p) - windkessel_residual(L - h, p)) / 2h rtol = 1e-7
        pin = (2.0e-6, -3.0, sqrt(1.5 * eq.kappa))
        @test inflow_jacobian(L, pin) ≈ Trixi.ForwardDiff.derivative(l -> inflow_residual(l, pin), L) rtol = 1e-10
    end

    # u_b on the outgoing characteristic, the resistor relation with the numerical
    # mass flux, and that flux is what the functor returns.
    u_b = windkessel_exterior(bc, u_inner, 1, flux_es, eq)
    Q_J = flux_es(u_inner, u_b, 1, eq)[1]
    A_b, u_b_vel = cons2prim(u_b, eq)
    @test u_b_vel + 4 * wave_speed(A_b, eq) ≈ u_i + 4 * wave_speed(A_i, eq) rtol = 1e-14
    @test pressure(u_b, eq) - bc.p_C ≈ bc.R1 * Q_J rtol = 1e-6
    @test bc(u_inner, 1, 2, SVector(0.126), 0.0, flux_es, eq)[1] ≈ Q_J rtol = 1e-14

    # Newton tolerance 1e-6 in flow units, SI and CGS.
    @test abs(A_b / exact_root(bc, u_inner, eq) - 1) <= 1e-9
    eq_cgs = ArteryFlowEquations1D(1.06, 0.04, 2.0, 0.22038, 0.03, 7.0e6, 0.5)
    bc_cgs = BoundaryConditionWindkessel3Element(2487.5, 18697.0, 1.7529e-5; p_C0 = 3.5e4)
    u_cgs = SVector(1.07 * eq_cgs.A0, 1.07 * eq_cgs.A0 * 11.0)
    ub_cgs = windkessel_exterior(bc_cgs, u_cgs, 1, flux_es, eq_cgs)
    @test abs(ub_cgs[1] / exact_root(bc_cgs, u_cgs, eq_cgs) - 1) <= 1e-9
end

const L_C = 12.6
const A_D = 0.28274
const P_D = 1.0933e5
const EQ = ArteryFlowEquations1D(1.06, 0.04, 4 / 3, A_D, 0.03, 7.0e6, 0.5; p_d = P_D)
const A0_INIT = (sqrt(A_D) - P_D / ((4 / 3) * sqrt(pi) * 7.0e6 * 0.03 / A_D))^2
const R1, R2, C0 = 2487.5, 18697.0, 1.7529e-5
const T_C = 1.1

const AMP = (3.294, 1.9262, -1.4219, -0.66627, -0.33933, -0.37914, 0.22396, 0.1507,
             0.18735, 0.038625, 0.012643, -0.0042453, -0.012781, 0.014805, 0.012249,
             0.0076502, 0.0030692, -0.0012271, -0.0042581, -0.0069785, 0.0085652,
             0.0081881, 0.0056549, 0.0026358, -0.0050868, -0.0085829)
const PHASE = (-0.023974, -1.1801, 0.92701, -0.24118, -0.27471, -1.0557, 1.22, 1.0984,
               0.067483, 0.22262, -0.10093, -1.1044, -1.3739, 1.2797, 0.80827, 0.40757,
               0.195, -1.1371, -0.92102, -1.2364, 1.4539, 0.89599, 0.17623, -1.3003,
               -0.011056, -0.86463)
Q_carotid(t) = 6.5 + sum(AMP[m] * sin(2m * pi * (t + 0.055) / T_C + PHASE[m]) for m in 1:26)

function carotid_problem(; C = C0, Q_in = Q_carotid, p_C0 = 0.0, p_out = 0.0,
                         A_init = A0_INIT, tspan = (0.0, 1.0))
    bc = BoundaryConditionWindkessel3Element(R1, R2, C; p_out, p_C0)
    solver = DGSEM(LobattoLegendreBasis(4), flux_es, VolumeIntegralFluxDifferencing(flux_ec))
    mesh = TreeMesh(0.0, L_C, initial_refinement_level = 0, periodicity = false)
    semi = SemidiscretizationHyperbolic(mesh, EQ, (x, t, e) -> SVector(A_init, 0.0), solver;
                                        source_terms = source_terms_friction,
                                        boundary_conditions = (x_neg = BoundaryConditionInflowRiemann(Q_in),
                                                               x_pos = bc))
    return semidiscretize_windkessel(semi, bc, tspan; track_volume = true)
end

cfl_solve(ode; kw...) = solve(ode, RDPK3Sp510(); dt = 1e-4, adaptive = false,
                              save_everystep = false, callback = StepsizeCallback(cfl = 1.0), kw...)
fixed_solve(ode, dt; kw...) = solve(ode, RDPK3Sp510(); dt, adaptive = false,
                                    save_everystep = false, kw...)

# A periodic state after three cycles from rest, with the volume counters reset.
function spun_up(; C = C0)
    ode = carotid_problem(; C, tspan = (0.0, 3T_C))
    y = copy(cfl_solve(ode).u[end])
    y[ode.p.n + 2] = 0
    y[ode.p.n + 3] = 0
    return ode, y, 3T_C
end

function rel_err(a, b, w)
    ua = Trixi.wrap_array(dg_state(a, w), w.semi)
    ub = Trixi.wrap_array(dg_state(b, w), w.semi)
    return (A = maximum(abs, ua[1, :, :] .- ub[1, :, :]) / maximum(abs, ub[1, :, :]),
            Q = maximum(abs, ua[2, :, :] .- ub[2, :, :]) / maximum(abs, ub[2, :, :]),
            p_C = abs(a[w.n + 1] - b[w.n + 1]) / abs(b[w.n + 1]))
end

function lsq_slope(x, y)
    lx, ly = log.(x), log.(y)
    return sum((lx .- sum(lx) / length(lx)) .* (ly .- sum(ly) / length(ly))) /
           sum(abs2, lx .- sum(lx) / length(lx))
end

@testset "coupled 1D-0D ODE" begin
    @testset "quiescent state" begin
        p_rest = pressure(SVector(A_D, 0.0), EQ)
        ode = carotid_problem(; Q_in = t -> 0.0, p_C0 = p_rest, p_out = p_rest, A_init = A_D)
        dy = similar(ode.u0)
        windkessel_rhs!(dy, ode.u0, ode.p, 0.0)
        @test maximum(abs, dy) <= 1e-10
    end

    @testset "steady RCR limit" begin
        Q0 = 5.0
        tau = R2 * C0
        p0 = 0.9 * Q0 * R2
        A_start = A_D * (1 + (p0 - P_D) / EQ.K_p)^2
        ode = carotid_problem(; Q_in = t -> Q0, p_C0 = p0, A_init = A_start, tspan = (0.0, 40tau))
        @test abs(cfl_solve(ode).u[end][ode.p.n + 1] / (Q0 * R2) - 1) <= 1e-8
    end

    ode, y0, t0 = spun_up()
    w = ode.p

    @testset "purity" begin
        dy1 = similar(y0)
        dy2 = similar(y0)
        windkessel_rhs!(dy1, y0, w, t0)
        w.bc.p_C = -1.0e9
        windkessel_rhs!(dy2, y0, w, t0)
        @test dy1 == dy2
        @test w.bc.p_C == y0[w.n + 1]
    end

    @testset "flux identity at every stage" begin
        # The coupled RHS reads Q_J and the inlet volume rate from Trixi's stored
        # boundary fluxes; re-evaluating the functors must agree to a few ulp.
        mismatch = Ref(0.0)
        bc_in = w.semi.boundary_conditions.x_neg
        function rhs_checked!(dy, y, w, t)
            windkessel_rhs!(dy, y, w, t)
            u = Trixi.wrap_array(dg_state(y, w), w.semi)
            u_out = SVector(u[1, end, w.element_out], u[2, end, w.element_out])
            u_in = SVector(u[1, 1, w.element_in], u[2, 1, w.element_in])
            F_out = w.bc(u_out, 1, 2, SVector(L_C), t, flux_es, EQ)[1]
            F_in = bc_in(u_in, 1, 1, SVector(0.0), t, flux_es, EQ)[1]
            Q_J = w.bc.C * dy[w.n + 1] + (y[w.n + 1] - w.bc.p_out) / w.bc.R2
            mismatch[] = max(mismatch[], abs(F_out - Q_J) / abs(Q_J),
                             abs(F_in - dy[w.n + 2]) / abs(F_in))
        end
        cfl_solve(ODEProblem(rhs_checked!, copy(y0), (t0, t0 + T_C), w))
        @test mismatch[] <= 1e-12
    end

    @testset "volume balance" begin
        prob = remake(ode; u0 = copy(y0), tspan = (t0, t0 + T_C))
        sol = cfl_solve(prob; saveat = t0:0.01:(t0 + T_C))
        @test volume_balance(sol, w).relative <= 1e-12
    end

    @testset "temporal order 5 (C = $C)" for C in (C0, C0 / 10)
        odeC, yC, tC = C == C0 ? (ode, y0, t0) : spun_up(; C)
        prob = remake(odeC; u0 = copy(yC), tspan = (tC, tC + T_C))
        ref = fixed_solve(prob, T_C / 40000).u[end]
        ns = (330, 400, 480, 576, 690, 830, 1000)
        errs = [rel_err(fixed_solve(prob, T_C / n).u[end], ref, odeC.p) for n in ns]
        dts = [T_C / n for n in ns]
        for q in (:A, :Q, :p_C)
            e = getfield.(errs, q)
            keep = e .> 1e-12
            s = lsq_slope(dts[keep], e[keep])
            @printf("  C = %.3e, %-3s: slope %.2f\n", C, q, s)
            @test count(keep) >= 4
            @test s >= 5.0
        end
    end

    @testset "saved values are step values with tstops" begin
        grid = collect(range(t0, t0 + 0.32, step = 0.001))
        prob = remake(ode; u0 = copy(y0), tspan = (t0, t0 + 0.32))
        ref = fixed_solve(prob, 0.001 / 40; saveat = grid)
        on = cfl_solve(prob; saveat = grid, tstops = grid)
        off = cfl_solve(prob; saveat = grid)
        err_on = maximum(rel_err(on.u[k], ref.u[k], w).Q for k in eachindex(grid))
        err_off = maximum(rel_err(off.u[k], ref.u[k], w).Q for k in eachindex(grid))
        @test err_on <= 1e-8
        @test err_off > 100 * err_on
    end
end
