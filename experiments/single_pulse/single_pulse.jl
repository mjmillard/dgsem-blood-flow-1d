# Single pulse in a reflection-free vessel (Boileau et al. 2015, Section 4.1,
# Table II, Eqs. (14)-(15)), thesis Section 5.1. A Gaussian inflow pulse travels
# along a 10 m vessel with an absorbing outlet, once inviscid (the crest should
# keep its amplitude) and once viscous (the crest should follow the Eq. (15)
# envelope). Writes results/single_pulse/single_pulse_pressure.png,
# single_pulse_analytic_errors.md and single_pulse_errors.md.
using Trixi
using OrdinaryDiffEqLowStorageRK
using Plots
using LaTeXStrings
using BloodFlowDGSEM

# Table II in CGS. nu = 0.5 gives β = (4/3)√π E h0, the paper's tube law, and
# ζ = 9 gives α = 1.1, for which K_R = 2π μ (ζ + 2)/ρ as in Eq. (15).
const SP_ZETA = 9
const SP_ALPHA = (SP_ZETA + 2) / (SP_ZETA + 1)
const SP_L = 1000.0
const SP_A0 = pi
const SP_RHO = 1.05
const SP_MU_VISC = 0.04
const SP_EQ_INV = ArteryFlowEquations1D(SP_RHO, 0.0, SP_ALPHA, SP_A0, 0.15, 4.0e6, 0.5)
const SP_EQ_VISC = ArteryFlowEquations1D(SP_RHO, SP_MU_VISC, SP_ALPHA, SP_A0, 0.15, 4.0e6, 0.5)
const SP_C0 = wave_speed(SP_A0, SP_EQ_INV)    # √(K_p / 2ρ) = 6.17 m/s
const SP_TSPAN = (0.0, 1.6)
const SP_SNAPSHOT_TIMES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

"""
    run_single_pulse(equations; refine = 8, cfl = 2.0)

N = 4, RDPK3Sp510 at the given CFL, `2^refine` elements. Saves every 5 ms with
`tstops` on the same grid, so each saved state is an RK step value. Returns the
solution, the semidiscretization and the CFL steps proposed at each step.
"""
function run_single_pulse(equations; refine = 8, cfl = 2.0)
    Q_in(t) = 1.0 * exp(-10000.0 * (t - 0.05)^2)    # Eq. (14): 1e-6 m³/s = 1 cm³/s
    boundary_conditions = (x_neg = BoundaryConditionInflowRiemann(Q_in),
                           x_pos = BoundaryConditionOutflowNonReflecting(SVector(SP_A0, 0.0)))
    initial_condition(x, t, equations) = SVector(equations.A0, zero(equations.A0))
    solver = DGSEM(LobattoLegendreBasis(4), flux_es, VolumeIntegralFluxDifferencing(flux_ec))
    mesh = TreeMesh(0.0, SP_L, initial_refinement_level = refine, periodicity = false)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        source_terms = source_terms_friction,
                                        boundary_conditions = boundary_conditions)
    ode = semidiscretize(semi, SP_TSPAN)

    saveat_times = collect(range(SP_TSPAN[1], SP_TSPAN[2], step = 0.005))
    dt_callback, dt_cfl = cfl_dt_recorder()
    callbacks = CallbackSet(AnalysisCallback(semi, interval = 100),
                            StepsizeCallback(cfl = cfl), dt_callback)
    sol = solve(ode, RDPK3Sp510(); dt = 1, adaptive = false, ode_default_options()...,
                callback = callbacks, saveat = saveat_times, tstops = saveat_times)
    return sol, semi, dt_cfl
end

# Gauge pressure K_p (√(A/A0) - 1) at the points x [cm], from the DG polynomial.
pressure_at(u, semi, equations, x) =
    equations.K_p .* (sqrt.(evaluate_dg(u, semi, x) ./ equations.A0) .- 1)

const X_PROFILE = collect(range(0.0, SP_L, step = 0.1))   # 1 mm grid
pressure_profile(u, semi, equations) = (X_PROFILE, pressure_at(u, semi, equations, X_PROFILE))

# Peak inlet pressure over the run.
inlet_peak(sol, semi, equations) =
    maximum(pressure_at(u, semi, equations, [0.0])[1] for u in sol.u)

# Crest (x, p) of a sampled pulse: the parabola through the largest sample and its
# neighbours. On the codes' 1-10 mm grids the largest sample alone sits up to
# 0.3 % below the crest of the ~4 cm wide pulse.
function crest(x, p)
    i = clamp(argmax(p), 2, length(p) - 1)
    y1, y2, y3 = p[i - 1], p[i], p[i + 1]
    d = (y1 - y3) / (2 * (y1 - 2y2 + y3))
    return x[i] + d * (x[i + 1] - x[i]), y2 - (y1 - y3) * d / 4
end

# Eq. (15): P_visc(x) / P_inv = exp(-(ζ + 2) π μ x / (ρ c0 A0)), x in cm.
env_norm(x_cm) = exp(-(SP_ZETA + 2) * pi * SP_MU_VISC * x_cm / (SP_RHO * SP_C0 * SP_A0))

# OLS fit z = v t + b with a free intercept.
function fit_speed(t, z)
    tm, zm = sum(t) / length(t), sum(z) / length(z)
    v = sum((t .- tm) .* (z .- zm)) / sum(abs2, t .- tm)
    return v, zm - v * tm
end

function main()
    outdir = results_dir("single_pulse")
    thesis_plot_style!()
    refine = 8    # 256 elements
    sol_inv, semi_inv, dt_cfl_inv = run_single_pulse(SP_EQ_INV; refine)
    sol_visc, semi_visc, _ = run_single_pulse(SP_EQ_VISC; refine)
    P_peak = inlet_peak(sol_inv, semi_inv, SP_EQ_INV)

    # Figure: p / p_peak against z at the snapshot times, inviscid solid, viscous
    # dashed, crests labelled with their time, and the Eq. (15) envelope.
    crest_colors = ["#2F6FC0", "#D6402E", "#2F9E63", "#E5732A",
                    "#6E4CC4", "#2AA7C4", "#C63A93", "#CDA01B"]
    plt = plot(size = (1200, 720), legend = :outerbottom, legend_columns = -1,
               xlabel = L"z" * "  [m]\n ", ylabel = L"p / p_{\mathrm{peak}}",
               guidefontsize = 13, tickfontsize = 11, legendfontsize = 11,
               ylims = (-0.03, 1.13), left_margin = 8 * Plots.mm,
               right_margin = 8 * Plots.mm, top_margin = 6 * Plots.mm,
               bottom_margin = 3 * Plots.mm, framestyle = :box)

    inv_crest_x = Float64[]
    inv_crest_val = Float64[]
    visc_crest_val = Float64[]
    visc_crest_x = Float64[]
    for (i, t) in enumerate(SP_SNAPSHOT_TIMES)
        x_inv, P_inv = pressure_profile(sol_inv(t), semi_inv, SP_EQ_INV)
        x_visc, P_visc = pressure_profile(sol_visc(t), semi_visc, SP_EQ_VISC)
        Pn_inv = P_inv ./ P_peak
        Pn_visc = P_visc ./ P_peak

        col = crest_colors[i]
        plot!(plt, x_inv ./ 100, Pn_inv; color = col, linewidth = 1.2, linestyle = :solid,
              label = "")
        plot!(plt, x_visc ./ 100, Pn_visc; color = col, linewidth = 1.6, linestyle = :dash,
              label = "")

        xc_inv, Pc_inv = crest(x_inv, Pn_inv)
        push!(inv_crest_x, xc_inv)
        push!(inv_crest_val, Pc_inv)
        annotate!(plt, xc_inv / 100, 1.07, text("t=$(t)s", TICKFONT, col, :center))

        xc_visc, Pc_visc = crest(x_visc, Pn_visc)
        push!(visc_crest_val, Pc_visc)
        push!(visc_crest_x, xc_visc)
    end

    x_env_cm = collect(range(0.0, SP_L, length = 400))
    plot!(plt, x_env_cm ./ 100, env_norm.(x_env_cm);
          color = :black, linewidth = 3, linestyle = :solid, label = "envelope")
    # Neutral legend entries, not tied to a snapshot colour.
    plot!(plt, [NaN], [NaN]; color = :gray40, linewidth = 1.2, linestyle = :solid,
          label = "inviscid")
    plot!(plt, [NaN], [NaN]; color = :gray40, linewidth = 1.6, linestyle = :dash,
          label = "viscous")
    thesis_savefig(plt, joinpath(outdir, "single_pulse_pressure.png"))

    # Comparison with the analytic predictions, for ours and the six codes:
    #   inviscid crest change P_crest(1.5 s) / P_crest(0.1 s) - 1;
    #   viscous crest / inviscid crest at the same time against Eq. (15), the
    #   signed deviation of largest magnitude over the snapshots;
    #   inviscid speed from an OLS fit of the crest positions, against c0;
    #   inviscid crest against the inlet peak (ours only).
    crests = Dict("present" => (inv_crest_x, inv_crest_val, visc_crest_x, visc_crest_val))
    bench_inv = load_benchmark_methods(data_dir("single_pulse"); suffix = "_inviscid")
    bench_visc = load_benchmark_methods(data_dir("single_pulse"); suffix = "_viscous")
    methods = ("DCG", "LCG", "FEM", "FVM", "McC", "STM")
    for m in methods
        ci = [crest(100 .* bench_inv[m][:, 1], bench_inv[m][:, i + 1])
              for i in eachindex(SP_SNAPSHOT_TIMES)]
        cv = [crest(100 .* bench_visc[m][:, 1], bench_visc[m][:, i + 1])
              for i in eachindex(SP_SNAPSHOT_TIMES)]
        crests[m] = (first.(ci), last.(ci), first.(cv), last.(cv))
    end

    analytic = Dict{String, NamedTuple}()
    for (s, (_, P_i, x_v, P_v)) in crests
        dev = P_v ./ P_i ./ env_norm.(x_v) .- 1
        analytic[s] = (change = P_i[end] / P_i[1] - 1, envelope = dev[argmax(abs.(dev))])
    end

    t0 = 0.05    # peak of Q_in
    inlet_dev = inv_crest_val .- 1
    i_inlet = argmax(abs.(inlet_dev))
    solutions = ["present", methods...]
    speed_rows = NamedTuple[]
    for s in solutions
        v_fit, b = fit_speed(SP_SNAPSHOT_TIMES, crests[s][1])
        push!(speed_rows, (solution = s, v_fit = v_fit, c0 = SP_C0, dv = v_fit / SP_C0 - 1,
                           t_launch = -b / v_fit, dt_launch = -b / v_fit - t0))
    end
    inlet_rows = NamedTuple[s == "present" ?
        (solution = s, dev = abs(inlet_dev[i_inlet]), t = SP_SNAPSHOT_TIMES[i_inlet]) :
        (solution = s, dev = nothing, t = nothing) for s in solutions]

    emit_tables([TableSpec("Crest vs the analytic prediction",
        [TableColumn(:metric, "metric"; fmt = :str),
         (TableColumn(Symbol(s), s; fmt = :pctsigned, digits = 3) for s in solutions)...],
        NamedTuple[(; metric, (Symbol(s) => analytic[s][key] for s in solutions)...)
                   for (metric, key) in (("inviscid crest change", :change),
                                         ("viscous vs envelope", :envelope))];
        notes = ["Inviscid crest change = P_crest(1.5 s) / P_crest(0.1 s) - 1 (> 0: the crest grew); " *
                 "the paper reports < 0.9 % in magnitude for all six codes.",
                 "Viscous vs envelope = P_crest,visc / P_crest,inv / exp(-k x_crest,visc) - 1 at the " *
                 "same snapshot (envelope: Boileau et al. Eq. (15)), signed value of largest magnitude " *
                 "over the 8 snapshots.",
                 "Crests by a parabola through the largest sample and its neighbours, on each code's " *
                 "own grid; ours from the DG solution on a 1 mm grid."]),
        TableSpec("Inviscid propagation speed (crest-position fit)",
            [TableColumn(:solution, "solution"; fmt = :str),
             TableColumn(:v_fit, "v_fit [cm/s]"; fmt = :fixed, digits = 2),
             TableColumn(:c0, "c0 [cm/s]"; fmt = :fixed, digits = 2),
             TableColumn(:dv, "v_fit / c0 - 1"; fmt = :pctsigned, digits = 3),
             TableColumn(:t_launch, "t_launch [s]"; fmt = :fixed, digits = 5),
             TableColumn(:dt_launch, "t_launch - t0 [s]"; fmt = :fixed, digits = 5)],
            speed_rows;
            notes = ["v_fit, b: ordinary least squares z_crest(t) = v_fit t + b with a free intercept over " *
                     "the 8 inviscid crests (z in cm, t in s).",
                     "c0 = sqrt(K_p / (2 rho)), the linear wave speed at A0 (CGS).",
                     "t_launch = -b / v_fit, the effective launch time; t0 = 0.05 s, the peak of the inlet Q_in (Eq. (14)).",
                     "Deviations from the linear analytic prediction (speed c0, launch at t0), not pure " *
                     "discretization errors: nonlinear convection moves the crest at U + c > c0."]),
        TableSpec("Inviscid crest vs the inlet peak",
            [TableColumn(:solution, "solution"; fmt = :str),
             TableColumn(:dev, "max |P_crest / P_peak - 1|"; fmt = :pct, digits = 3),
             TableColumn(:t, "at t [s]"; fmt = :fixed, digits = 1)],
            inlet_rows;
            notes = ["P_peak = max over the run of the inviscid inlet pressure P(0, t); maximum over the " *
                     "8 snapshots, with the snapshot time at which it occurs.",
                     "Deviation from the linear analytic prediction (crest keeps its inlet amplitude), " *
                     "not a pure discretization error. Unlike the first-to-last crest change above, it is " *
                     "referenced to the inlet, not the first snapshot.",
                     "Not available for the six codes: the published data hold only x and P at the 8 " *
                     "snapshots, no inlet pressure (—)."])],
        outdir, "single_pulse_analytic_errors";
        title = "Single pulse -- crest vs the analytic prediction (Boileau et al. Section 4.1)")

    # Resolution as in Boileau et al. Table I, from the inviscid run: mean
    # accepted dt, mean CFL-proposed dt (steps are also clipped to the 5 ms grid),
    # and CFL = λ_max dt / Omega_e over the saved states.
    h_elem = SP_L / 2^refine
    lam_max = maximum(maximum(Trixi.max_abs_speeds(SVector(u[1, i, e], u[2, i, e]),
                                                   SP_EQ_INV)[1]
                              for i in axes(u, 2), e in axes(u, 3))
                      for u in (Trixi.wrap_array(v, semi_inv) for v in sol_inv.u))
    dt_mean = (SP_TSPAN[2] - SP_TSPAN[1]) / sol_inv.stats.naccept
    dt_cfl_mean = sum(dt_cfl_inv) / length(dt_cfl_inv)
    resolution = (dt = 1e3 * dt_mean, h = h_elem, cfl = lam_max * dt_mean / h_elem,
                  space = "5th", time = "5th", polydeg = 4,
                  dt_cfl = 1e3 * dt_cfl_mean, cfl_cfl = lam_max * dt_cfl_mean / h_elem)
    emit_tables([resolution_table("single_pulse", resolution)], outdir, "single_pulse_errors";
                title = "Single pulse -- numerical parameters (Boileau et al., Table I)")

    println("c0 = $(round(SP_C0 / 100, digits = 4)) m/s, P_peak = $(round(P_peak, digits = 4)) dyn/cm^2")
    println("inviscid crest change $(round(100 * analytic["present"].change, digits = 3)) %, " *
            "viscous vs envelope $(round(100 * analytic["present"].envelope, digits = 3)) %")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
