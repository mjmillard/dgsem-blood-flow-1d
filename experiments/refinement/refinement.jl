# Discretization checks of the thesis appendix "Benchmark Comparison Details"
# (section Discretization Refinement: its table and the time-step and mesh
# sensitivity statements).
# Deviations are taken against finer DGSEM solutions: 64 elements for the carotid
# and aorta, normalized as in E_MAX; 2048 elements for the single pulse, the
# largest pressure difference over the eight snapshots divided by p_peak.
# Δt is the mean CFL step. Wall times are machine dependent and not reproduced.
# Writes results/refinement/refinement.md. About 15 minutes on one core.
using Printf
using BloodFlowDGSEM
using BloodFlowDGSEM: boileau_errors, BENCHMARK_POINTWISE, BENCHMARK_QUANTITIES, ERROR_METRICS

include(joinpath(@__DIR__, "..", "carotid", "carotid.jl"))
include(joinpath(@__DIR__, "..", "aorta", "aorta.jl"))
include(joinpath(@__DIR__, "..", "single_pulse", "single_pulse.jl"))

# Largest normalized deviation of each midpoint quantity from the reference, per cent.
max_deviation(run, ref_run) =
    map((a, b, pw) -> 100 * boileau_errors(a, b; pointwise = pw).max,
        run.ours, ref_run.ours, BENCHMARK_POINTWISE)

# The present errors against 3D, per cent, keyed by (metric, quantity).
errors_vs_3d(run) =
    Dict((k, q) => 100 * getfield(boileau_errors(run.ours[q], run.ref[q];
                                                 pointwise = BENCHMARK_POINTWISE[q]), k)
         for (k, _) in ERROR_METRICS, q in BENCHMARK_QUANTITIES)

function max_change(run_a, run_b; metrics = first.(ERROR_METRICS))
    Ea, Eb = errors_vs_3d(run_a), errors_vs_3d(run_b)
    return maximum(abs(Ea[(k, q)] - Eb[(k, q)]) for k in metrics, q in BENCHMARK_QUANTITIES)
end

function arterial_table(case, refines; ref_refine = 6)
    ref_run = run_benchmark(case; refine = ref_refine)
    rows = NamedTuple[]
    runs = Dict{Int, Any}()
    for r in refines
        run = runs[r] = run_benchmark(case; refine = r)
        d = max_deviation(run, ref_run)
        push!(rows, (elements = 2^r, h = run.h_elem, dt = 1e3 * run.dt_mean,
                     p = d.p, Q = d.Q, dr = d.dr, dp = d.dp))
    end
    rms = errors_vs_3d(runs[case.refine])
    spec = TableSpec("$(case.name): maximum normalized deviation from $(2^ref_refine) elements (per cent)",
        [TableColumn(:elements, "elements"; fmt = :int),
         TableColumn(:h, "Δz_e [cm]"; fmt = :fixed, digits = 3),
         TableColumn(:dt, "Δt [ms]"; fmt = :fixed, digits = 2),
         TableColumn(:p, "p"; fmt = :sci, digits = 2),
         TableColumn(:Q, "Q"; fmt = :sci, digits = 2),
         TableColumn(:dr, "Δr"; fmt = :sci, digits = 2),
         TableColumn(:dp, "Δp"; fmt = :sci, digits = 2)],
        rows;
        notes = ["N = 4, RDPK3Sp510, CFL 2.0; Δt is the mean accepted (CFL) step.",
                 "E_RMS against 3D at the production mesh (p/Q/Δr/Δp): " *
                 join((@sprintf("%.2f", rms[(:rms, q)]) for q in BENCHMARK_QUANTITIES), "/") * "."])
    return spec, runs
end

function single_pulse_table(refines; ref_refine = 11)
    profiles(sol, semi, eq) = [pressure_profile(sol(t), semi, eq)[2] for t in SP_SNAPSHOT_TIMES]
    sol_ref, semi_ref, _ = run_single_pulse(SP_EQ_INV; refine = ref_refine)
    P_peak = inlet_peak(sol_ref, semi_ref, SP_EQ_INV)
    ref_inv = profiles(sol_ref, semi_ref, SP_EQ_INV)
    sol_ref, semi_ref, _ = run_single_pulse(SP_EQ_VISC; refine = ref_refine)
    ref_visc = profiles(sol_ref, semi_ref, SP_EQ_VISC)

    deviation(p, ref) = 100 * maximum(maximum(abs, a .- b) for (a, b) in zip(p, ref)) / P_peak
    rows = NamedTuple[]
    for r in refines
        sol_inv, semi_inv, dt_cfl = run_single_pulse(SP_EQ_INV; refine = r)
        sol_visc, semi_visc, _ = run_single_pulse(SP_EQ_VISC; refine = r)
        p_inv = profiles(sol_inv, semi_inv, SP_EQ_INV)
        crests = [crest(X_PROFILE, p ./ P_peak)[2] for p in p_inv]
        push!(rows, (elements = 2^r, h = SP_L / 2^r, dt = 1e3 * sum(dt_cfl) / length(dt_cfl),
                     inviscid = deviation(p_inv, ref_inv),
                     viscous = deviation(profiles(sol_visc, semi_visc, SP_EQ_VISC), ref_visc),
                     crest_change = crests[end] / crests[1] - 1))
    end
    return TableSpec("single pulse: maximum pressure deviation from $(2^ref_refine) elements over the 8 snapshots (per cent of p_peak)",
        [TableColumn(:elements, "elements"; fmt = :int),
         TableColumn(:h, "Δz_e [cm]"; fmt = :fixed, digits = 2),
         TableColumn(:dt, "Δt [ms]"; fmt = :fixed, digits = 2),
         TableColumn(:inviscid, "inviscid"; fmt = :sci, digits = 2),
         TableColumn(:viscous, "viscous"; fmt = :sci, digits = 2),
         TableColumn(:crest_change, "inviscid crest change"; fmt = :pctsigned, digits = 3)],
        rows;
        notes = ["N = 4, RDPK3Sp510, CFL 2.0; Δt is the mean step proposed by the CFL condition.",
                 "Crest change = P_crest(1.5 s) / P_crest(0.1 s) - 1, on the 1 mm grid."])
end

function main()
    outdir = results_dir("refinement")
    carotid_spec, carotid_runs = arterial_table(CAROTID, (0, 1))
    aorta_spec, aorta_runs = arterial_table(AORTA, (0, 1, 2, 3))
    single_spec = single_pulse_table(6:10)

    # Sensitivity of the 3D comparison (percentage points): CFL 0.25 against 2.0 at
    # the production meshes, and four against two aortic elements.
    checks = NamedTuple[]
    for (case, runs) in ((CAROTID, carotid_runs), (AORTA, aorta_runs))
        run_cfl = run_benchmark(case; cfl = 0.25)
        push!(checks, (check = "$(case.name): CFL 0.25 vs 2.0",
                       all = max_change(run_cfl, runs[case.refine]),
                       rms = max_change(run_cfl, runs[case.refine]; metrics = (:rms,))))
    end
    push!(checks, (check = "aorta: 4 vs 2 elements",
                   all = max_change(aorta_runs[2], aorta_runs[1]),
                   rms = max_change(aorta_runs[2], aorta_runs[1]; metrics = (:rms,))))
    check_spec = TableSpec("Sensitivity of the errors against 3D (percentage points)",
        [TableColumn(:check, "comparison"; fmt = :str),
         TableColumn(:all, "max change, all entries"; fmt = :sci, digits = 2),
         TableColumn(:rms, "max change, E_RMS"; fmt = :sci, digits = 2)],
        checks; notes = ["Over the four metrics and four quantities of the 3D error table."])

    emit_tables([carotid_spec, aorta_spec, single_spec, check_spec], outdir, "refinement";
                title = "Discretization refinement")
end

main()
