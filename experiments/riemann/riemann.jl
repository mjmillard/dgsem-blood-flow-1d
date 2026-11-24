# Riemann problems of Toro (2025), Chapter 9, Table 9.2, against the exact solution
# (thesis Section 4.2 and the appendix on the Riemann tests). With N = 4 on 512 cells and
# Hennemann-Gassner shock capturing, all four tests are run at one indicator
# threshold T, chosen by a sweep over a T grid.
#
#     julia --project experiments/riemann/riemann.jl             # T = T_SELECTED
#     julia --project experiments/riemann/riemann.jl --T-sweep   # redo the sweep
#
# Writes results/riemann/test{1..4}_{area,velocity,area_error,velocity_error,
# xt_diagram,shock_indicator}.png and riemann_errors.md; with --T-sweep also
# test{1..4}_T_sweep.png, T_sweep_all.png and riemann_T_sweep.md.
using Trixi
using OrdinaryDiffEqLowStorageRK
using Plots
using LaTeXStrings
using Printf
using BloodFlowDGSEM

include("exact_riemann_solver.jl")

thesis_plot_style!()

const OUTDIR = results_dir("riemann")
out(name) = joinpath(OUTDIR, name)

# Table 9.1 in CGS. The chapter is inviscid with α = 1 (no source terms); ν = 1/2
# makes the code's κ equal Toro's Eq. (9.5).
const equations = ArteryFlowEquations1D(1.0, 0.0, 1.0, 3.14, 0.05, 3.0e6, 0.5)
const KAPPA = equations.kappa
const A0 = 3.14
@assert isapprox(KAPPA, toro_kappa(E = 3.0e6, h0 = 0.05, rho = 1.0, A0 = 3.14); rtol = 1e-12)

const POLYDEG = 4
const REFINEMENT_LEVEL = 9    # 512 cells

# The threshold the sweep selects (the grid point 10^-5.5; see riemann_T_sweep.md).
const T_SELECTED = exp10(-5.5)

# CFL. The blended finite-volume subcells at the element ends are w_0 Δx/2 = Δx/20
# wide, while StepsizeCallback scales the step to the mean subcell Δx/(N+1), so
# they run at four times the nominal CFL. The subcritical tests are stable to CFL
# 1.6, but L2(A) grows by 7 % already at 1.25, so they use 1.0. Test 2 is
# supercritical and blows up at 0.75 (A < 0), so it uses 0.6.
const CFL_SUB = 1.0
const CFL_SUPER = 0.6
const BASE_LENGTH = 50.0
const REACH_SAFETY = 0.8      # widen the domain if a wave gets this close to an edge

# Ceiling on the blending factor θ, the same for all four tests.
const ALPHA_MAX = 0.5

# Largest |u|/c anywhere in the exact solution.
function max_froude(rsol::RiemannSolution; n = 20001)
    m = 0.0
    span = 3 * max_signal_speed(rsol)
    for xi in range(-span, span, length = n)
        A, u = sample(rsol, xi)
        m = max(m, abs(u) / c_wave(A, rsol.kappa))
    end
    return m
end

cfl_for(rsol::RiemannSolution) = max_froude(rsol) > 1 ? CFL_SUPER : CFL_SUB

# The T grid: half decades over [1e-10, 1], quarter decades over [1e-7, 1e-4]
# where the overshoot cap binds, plus the threshold T(4) ≈ 1.02e-3 of Hennemann
# et al. (2021). Deduplicated on the rounded exponent, since the two ranges share
# endpoints.
const T_PAPER = hg_paper_threshold(LobattoLegendreBasis(POLYDEG))
const T_EXPONENTS = sort(unique(round.(vcat(range(-10, 0, length = 21),
                                            range(-7, -4, length = 13)); digits = 6)))
const T_SWEEP = sort(vcat(exp10.(T_EXPONENTS), [T_PAPER]))

# Exact solution and a domain in which no wave reaches a boundary by Tout (so the
# exact Dirichlet data at both ends are the initial states). The domain is
# widened in 10 cm steps until the outermost wave stays within REACH_SAFETY of
# the edge; this widens Test 2 to L = 60 cm.
function configure_test(test)
    L = BASE_LENGTH
    while true
        x0 = L / 2
        rsol = RiemannSolution(test.AL, test.uL, test.AR, test.uR, x0, KAPPA)
        reach = wave_reach_speed(rsol) * test.Tout
        if reach <= REACH_SAFETY * min(x0, L - x0)
            return (L = L, x0 = x0, rsol = rsol, reach = reach,
                    margin = min(x0, L - x0) - reach)
        end
        L += 10.0
    end
end

function make_initial_condition(rsol::RiemannSolution)
    return function initial_condition_riemann(x, t, equations::ArteryFlowEquations1D)
        A, u = solution_at(rsol, x[1], t)
        return SVector(A, A * u)
    end
end

# Volume-normalized L1 error of func(u) on the analyzer's nodes. Adapted from
# Trixi.jl's `calc_error_norms` for TreeMesh{1} (MIT License, Copyright (c)
# 2020-present The Trixi.jl Authors), accumulating |diff| instead of diff².
function calc_l1_error(func, u_ode, t, analyzer, semi, cache_analysis)
    mesh, eqs, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, mesh, eqs, dg, cache)
    ic = semi.initial_condition
    vandermonde = analyzer.vandermonde
    weights = analyzer.weights
    node_coordinates = cache.elements.node_coordinates
    u_local = cache_analysis.u_local
    x_local = cache_analysis.x_local

    l1 = zero(func(Trixi.get_node_vars(u, eqs, dg, 1, 1), eqs))
    for element in Trixi.eachelement(dg, cache)
        Trixi.multiply_dimensionwise!(u_local, vandermonde, view(u, :, :, element))
        Trixi.multiply_dimensionwise!(x_local, vandermonde,
                                      view(node_coordinates, :, :, element))
        volume_jacobian_ = Trixi.volume_jacobian(element, mesh, cache)
        for i in Trixi.eachnode(analyzer)
            u_exact = ic(Trixi.get_node_coords(x_local, eqs, dg, i), t, eqs)
            diff = func(u_exact, eqs) - func(Trixi.get_node_vars(u_local, eqs, dg, i), eqs)
            l1 += abs.(diff) * (weights[i] * volume_jacobian_)
        end
    end
    return l1 ./ Trixi.total_volume(mesh)
end

# Oscillation measures on the output profile.
#   overshoot: largest excursion of A outside the two states a shock connects,
#     within 12 cells of the exact shock, as a fraction of the exact A range (the
#     figure's height). A monotone smeared shock gives zero. This is the quantity
#     the threshold selection caps at OVERSHOOT_TOL.
#   fan_excursion: largest departure from a local ±4-cell envelope of the exact
#     rarefaction fans (staircasing from over-blending).
#   tv_excess: relative total-variation excess.
const WINDOW_CELLS = 4
const OVERSHOOT_TOL = 1.0e-2

function oscillation_metrics(x, A_num, rsol, t, dx; tol = OVERSHOOT_TOL,
                             window_cells = WINDOW_CELLS)
    A_exact, _ = exact_riemann_profile(rsol, x, t)
    n = length(x)
    global_range = maximum(A_exact) - minimum(A_exact)

    total_variation(v) = sum(abs, diff(v))
    tv_ex = total_variation(A_exact)
    tv_excess = (total_variation(A_num) - tv_ex) / tv_ex

    halfwidth = window_cells * dx

    overshoot = 0.0
    for (wave, head, s_in, s_out) in ((rsol.left_wave, rsol.SL_head, rsol.AL, rsol.A_star),
                                      (rsol.right_wave, rsol.SR_head, rsol.A_star, rsol.AR))
        wave === :shock || continue
        xc = rsol.x0 + head * t
        lo, hi = minmax(s_in, s_out)
        for i in eachindex(x)
            abs(x[i] - xc) <= 3 * halfwidth || continue
            overshoot = max(overshoot, max(0.0, A_num[i] - hi, lo - A_num[i]) / global_range)
        end
    end

    fan_excursion = 0.0
    for (wave, head, tail) in ((rsol.left_wave, rsol.SL_head, rsol.SL_tail),
                               (rsol.right_wave, rsol.SR_head, rsol.SR_tail))
        wave === :rarefaction || continue
        flo, fhi = minmax(rsol.x0 + head * t, rsol.x0 + tail * t)
        jlo, jhi = 1, 1
        for i in eachindex(x)
            (flo - halfwidth <= x[i] <= fhi + halfwidth) || continue
            while jlo < n && x[jlo] < x[i] - halfwidth
                jlo += 1
            end
            while jhi < n && x[jhi + 1] <= x[i] + halfwidth
                jhi += 1
            end
            elo, ehi = extrema(@view A_exact[jlo:jhi])
            fan_excursion = max(fan_excursion,
                                max(0.0, A_num[i] - ehi, elo - A_num[i]) / global_range)
        end
    end

    return (; tv_excess, overshoot, fan_excursion, no_overshoot = overshoot <= tol)
end

"""
    run_case(test, cfg, T; entropy_check = false, indicator_interval = 100)

One Table 9.2 case at threshold `T`: flux_ec volume flux, flux_es surface flux and
finite-volume flux, blended by `IndicatorHennemannGassnerT`, exact Dirichlet data
at both ends, RDPK3Sp510. Returns the error norms, the profile at Tout, the
oscillation measures and the indicator and entropy diagnostics.

The callback order is part of the result: the AnalysisCallback (interval 0, used
only for its analyzer) and the entropy-balance callback both evaluate the
right-hand side, which refreshes the indicator's cached θ that the shock monitor
records.
"""
function run_case(test, cfg, T; entropy_check = false, indicator_interval = 100)
    ic = make_initial_condition(cfg.rsol)
    basis = LobattoLegendreBasis(POLYDEG)
    indicator_sc = IndicatorHennemannGassnerT(equations, basis; alpha_max = ALPHA_MAX,
                                              alpha_min = 0.001, alpha_smooth = true,
                                              variable = area, T = T)
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc; volume_flux_dg = flux_ec,
                                                     volume_flux_fv = flux_es)
    solver = DGSEM(basis, flux_es, volume_integral)
    mesh = TreeMesh(0.0, cfg.L, initial_refinement_level = REFINEMENT_LEVEL,
                    periodicity = false)
    boundary_conditions = (x_neg = BoundaryConditionDirichlet(ic),
                           x_pos = BoundaryConditionDirichlet(ic))
    semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver;
                                        boundary_conditions = boundary_conditions)
    ode = semidiscretize(semi, (0.0, test.Tout))

    analysis_callback = AnalysisCallback(semi; interval = 0,
                                         analysis_errors = [:l2_error, :linf_error,
                                                            :l2_error_primitive,
                                                            :linf_error_primitive])
    shock_cb = ShockIndicatorMonitorCallback(semi; interval = indicator_interval)
    cbs = Any[analysis_callback, StepsizeCallback(cfl = cfl_for(cfg.rsol)), shock_cb]
    entropy_cb = nothing
    if entropy_check
        entropy_cb = EntropyBoundaryBalanceCallback(semi; surface_flux = flux_es,
                                                    boundary_condition_x_neg = boundary_conditions.x_neg,
                                                    boundary_condition_x_pos = boundary_conditions.x_pos,
                                                    interval = 100)
        push!(cbs, entropy_cb)
    end

    sol = solve(ode, RDPK3Sp510(); dt = 1, adaptive = false, ode_default_options()...,
                callback = CallbackSet(cbs...), save_everystep = false)

    af = analysis_callback.affect!
    u_end, t_end = sol.u[end], sol.t[end]
    l2_prim, linf_prim = Trixi.calc_error_norms(cons2prim, u_end, t_end, af.analyzer, semi,
                                                af.cache)
    l1_prim = calc_l1_error(cons2prim, u_end, t_end, af.analyzer, semi, af.cache)
    l2_cons, linf_cons = Trixi.calc_error_norms(cons2cons, u_end, t_end, af.analyzer, semi,
                                                af.cache)

    pd = PlotData1D(u_end, semi, solution_variables = cons2cons)
    x = pd.x
    A_num = pd.data[:, 1]
    u_num = pd.data[:, 2] ./ A_num
    osc = oscillation_metrics(x, A_num, cfg.rsol, t_end, cfg.L / 2^REFINEMENT_LEVEL;
                              tol = OVERSHOOT_TOL)

    ind = shock_indicator_history(shock_cb)
    frac_active = isempty(ind.n_active) ? 0.0 : maximum(ind.n_active) / ind.n_elements
    max_production = NaN
    if entropy_cb !== nothing
        eb = entropy_balance_history(entropy_cb)
        max_production = isempty(eb.production) ? NaN : maximum(eb.production)
    end

    return (; T, t_end,
            l1_A = l1_prim[1], l1_u = l1_prim[2],
            l2_A = l2_prim[1], l2_u = l2_prim[2],
            linf_A = linf_prim[1], linf_u = linf_prim[2],
            l2_Q = l2_cons[2], linf_Q = linf_cons[2],
            x, A_num, u_num,
            tv_excess = osc.tv_excess, overshoot = osc.overshoot,
            fan_excursion = osc.fan_excursion, no_overshoot = osc.no_overshoot,
            alpha_max = ALPHA_MAX, frac_active, max_production, shock_cb)
end

# Mutes the AnalysisCallback's initial printout.
run_case_quiet(args...; kwargs...) = redirect_stdout(() -> run_case(args...; kwargs...), devnull)

# `nothing` if the run diverges: an under-dissipated T can drive A negative, and
# A^{3/2} in flux_ec then throws a DomainError.
function try_run_case(args...; kwargs...)
    try
        return run_case_quiet(args...; kwargs...)
    catch err
        err isa DomainError || rethrow()
        return nothing
    end
end

# Wave pattern of the numerical solution: inside the window each exact wave must
# occupy (padded by two cells), a wave is a shock if the middle 80 % of its area
# change happens within four cells.
function classify_numerical(cfg, res)
    dx = cfg.L / (2^REFINEMENT_LEVEL)
    rsol = cfg.rsol
    t = res.t_end

    function classify(S_head, S_tail, A_out, A_in)
        lo = cfg.x0 + min(S_head, S_tail) * t - 2 * dx
        hi = cfg.x0 + max(S_head, S_tail) * t + 2 * dx
        idx = findall(xi -> lo <= xi <= hi, res.x)
        length(idx) < 3 && return "?"
        A = res.A_num[idx]
        target_lo = A_out + 0.1 * (A_in - A_out)
        target_hi = A_out + 0.9 * (A_in - A_out)
        between = findall(a -> min(target_lo, target_hi) <= a <= max(target_lo, target_hi), A)
        isempty(between) && return "S"
        width = res.x[idx[last(between)]] - res.x[idx[first(between)]]
        return width <= 4 * dx ? "S" : "R"
    end

    l = classify(rsol.SL_head, rsol.SL_tail, rsol.AL, rsol.A_star)
    r = classify(rsol.SR_tail, rsol.SR_head, rsol.A_star, rsol.AR)
    (l == "R" && rsol.left_wave === :rarefaction && rsol.SL_head < 0 < rsol.SL_tail) &&
        (l = "R(sonic)")
    return l * r
end

# ---------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------
const COL_NUM = "#2F6FC0"
const COL_EXACT = :black
const COL_SHOCK = "#D6402E"
const COL_FAN = "#2F6FC0"
const COL_ERROR = "#C1440E"

# Hollow circle as an explicit Shape: GR's symbol marker path cannot draw a
# transparent fill with an opaque stroke.
const MARKER_HOLLOW_CIRCLE = Shape(cos.(range(0, 2π; length = 72)),
                                   sin.(range(0, 2π; length = 72)))

# Number of markers for the numerical profile (thinned for legibility; it is not
# the resolution).
const N_PROFILE_MARKERS = 400

# Bottom of the log axis in plot_profile_error, relative to the field's magnitude.
const ERROR_FLOOR_REL = 1e-12

# Numerical (markers) vs exact (line on a fine grid) profile of A or u at Tout.
function plot_profile(test, cfg, res, field, filename; legend_pos = :best)
    x_fine = collect(range(0.0, cfg.L, length = 4000))
    A_ex, u_ex = exact_riemann_profile(cfg.rsol, x_fine, test.Tout)
    y_num, y_ex, ylabel = field === :A ? (res.A_num, A_ex, L"A" * "  [cm²]") :
                          (res.u_num, u_ex, L"u" * "  [cm/s]")

    plt = plot(size = (1000, 640), framestyle = :box,
               legend = field === :A ? legend_pos : false,
               xlabel = L"z" * "  [cm]", ylabel = ylabel,
               guidefontsize = 23, tickfontsize = 18, legendfontsize = 18,
               left_margin = 6 * Plots.mm, bottom_margin = 4 * Plots.mm)
    n = length(res.x)
    idx = unique(vcat(1:max(1, cld(n, N_PROFILE_MARKERS)):n, n))
    plot!(plt, x_fine, y_ex; color = COL_EXACT, linewidth = 2.0, label = "exact")
    scatter!(plt, res.x[idx], y_num[idx]; markershape = MARKER_HOLLOW_CIRCLE,
             markersize = 6, markercolor = RGBA(1, 1, 1, 0),
             markerstrokecolor = COL_NUM, markerstrokealpha = 1.0,
             markerstrokewidth = 1.3, label = "numerical (DGSEM)")
    thesis_savefig(plt, filename)
end

# log10 |numerical - exact| at the solution points; exact zeros are clamped to
# ERROR_FLOOR_REL times the field's magnitude.
function plot_profile_error(test, cfg, res, field, filename)
    A_ex, u_ex = exact_riemann_profile(cfg.rsol, res.x, test.Tout)
    y_num, y_ex, ylabel = field === :A ?
                          (res.A_num, A_ex, L"\log_{10}|A_{\mathrm{num}} - A_{\mathrm{exact}}|") :
                          (res.u_num, u_ex, L"\log_{10}|u_{\mathrm{num}} - u_{\mathrm{exact}}|")
    floor_ = log10(ERROR_FLOOR_REL * maximum(abs, y_ex))
    log_err = max.(log10.(abs.(y_num .- y_ex)), floor_)

    plt = plot(size = (1000, 640), framestyle = :box, legend = false,
               xlabel = L"z" * "  [cm]", ylabel = ylabel,
               ylims = (floor_, ceil(maximum(log_err))),
               yticks = 2fld(ceil(Int, maximum(log_err)), 2):-2:ceil(Int, floor_),
               guidefontsize = 23, tickfontsize = 18,
               left_margin = 6 * Plots.mm, bottom_margin = 4 * Plots.mm)
    plot!(plt, res.x, log_err; color = COL_ERROR, linewidth = 1.8)
    thesis_savefig(plt, filename)
end

# z-t diagram of the exact solution: fans as rays between head and tail, shocks
# as heavy lines. The contact is omitted (A and u are continuous across it).
function plot_xt_diagram(test, cfg, filename; n_rays = 15)
    rsol = cfg.rsol
    tmax = test.Tout
    ts = [0.0, tmax]
    ray(S) = cfg.x0 .+ S .* ts

    plt = plot(size = (1000, 640), framestyle = :box, legend = :best,
               background_color_legend = RGBA(1, 1, 1, 0.85), foreground_color_legend = :black,
               xlabel = L"z" * "  [cm]", ylabel = L"t" * "  [s]",
               guidefontsize = 23, tickfontsize = 18, legendfontsize = 18,
               xlims = (0.0, cfg.L), ylims = (0.0, tmax),
               left_margin = 6 * Plots.mm, bottom_margin = 4 * Plots.mm,
               right_margin = 5 * Plots.mm)

    if rsol.left_wave === :rarefaction
        for (k, S) in enumerate(range(rsol.SL_head, rsol.SL_tail, length = n_rays))
            plot!(plt, ray(S), ts; color = COL_FAN, linewidth = k in (1, n_rays) ? 2.0 : 0.8,
                  alpha = k in (1, n_rays) ? 1.0 : 0.55,
                  label = k == 1 ? "left rarefaction" : "")
        end
    else
        plot!(plt, ray(rsol.SL_head), ts; color = COL_SHOCK, linewidth = 3.0,
              label = "left shock")
    end
    if rsol.right_wave === :rarefaction
        for (k, S) in enumerate(range(rsol.SR_tail, rsol.SR_head, length = n_rays))
            plot!(plt, ray(S), ts; color = COL_FAN, linewidth = k in (1, n_rays) ? 2.0 : 0.8,
                  alpha = k in (1, n_rays) ? 1.0 : 0.55,
                  label = k == 1 ? "right rarefaction" : "")
        end
    else
        plot!(plt, ray(rsol.SR_head), ts; color = COL_SHOCK, linewidth = 3.0,
              label = "right shock")
    end
    vline!(plt, [cfg.x0]; color = :gray75, linestyle = :dot, linewidth = 1.0, label = L"z_0")
    thesis_savefig(plt, filename)
end

# Error norms of A and the overshoot against T for one test, with the cap, the
# paper's T(4) and the selected T.
function plot_T_sweep(sweep, filename, T_used; tol = OVERSHOOT_TOL, legend_pos = :bottomright)
    sweep = sort(sweep, by = r -> r.T)
    Ts = [r.T for r in sweep]
    plt = plot(size = (1000, 600), framestyle = :box, legend = legend_pos,
               xscale = :log10, yscale = :log10,
               xlabel = "HG threshold " * L"\mathbb{T}", ylabel = "error in " * L"A",
               guidefontsize = 17, tickfontsize = 14, legendfontsize = 13,
               left_margin = 2 * Plots.mm, bottom_margin = 6 * Plots.mm,
               right_margin = 6 * Plots.mm)
    plot!(plt, Ts, [r.l1_A for r in sweep]; marker = :circle, markersize = 3,
          color = "#2F6FC0", label = L"L^1")
    plot!(plt, Ts, [r.l2_A for r in sweep]; marker = :diamond, markersize = 3,
          color = "#D6402E", label = L"L^2")
    plot!(plt, Ts, [r.linf_A for r in sweep]; marker = :utriangle, markersize = 3,
          color = "#6E4CC4", label = L"L^\infty")
    plot!(plt, Ts, [max(r.overshoot, 1e-8) for r in sweep];
          marker = :star5, markersize = 5, linewidth = 2.5, color = "#2AA7C4",
          label = "overshoot")
    hline!(plt, [tol]; color = "#2AA7C4", linestyle = :dashdot, linewidth = 1.2,
           label = @sprintf("overshoot cap = %.0e", tol))
    vline!(plt, [T_PAPER]; color = :gray40, linestyle = :dash, linewidth = 1.5,
           label = latexstring("\\mathbb{T}($POLYDEG)") * @sprintf(" = %.2e", T_PAPER))
    vline!(plt, [T_used]; color = "#2F9E63", linewidth = 2.5,
           label = L"\mathbb{T}" * @sprintf(" selected = %.2e", T_used))
    thesis_savefig(plt, filename)
end

const SWEEP_TEST_STYLE = Dict(
    1 => (label = "Test 1 (RS)", color = "#2F6FC0", mark = :circle),
    2 => (label = "Test 2 (R(sonic)S)", color = "#D6402E", mark = :diamond),
    3 => (label = "Test 3 (RR)", color = "#6E4CC4", mark = :utriangle),
    4 => (label = "Test 4 (SS)", color = "#E5732A", mark = :star5))

# Floor for exact zeros on the log axis (Test 3 never overshoots).
const OVERSHOOT_FLOOR = 1e-8

# All four sweeps in one figure: L2(A) against T (left) and the overshoot against
# T with the cap (right).
function plot_T_sweep_combined(sweeps, filename, T_used; tol = OVERSHOOT_TOL)
    ids = sort(collect(keys(sweeps)))
    xticks = exp10.(-10:2:0)

    panel(; ylabel, legend, yticks = :auto, kwargs...) =
        plot(framestyle = :box, legend = legend,
             xscale = :log10, yscale = :log10, xticks = xticks, yticks = yticks,
             xlabel = "HG threshold " * L"\mathbb{T}", ylabel = ylabel,
             guidefontsize = 14, tickfontsize = 12, legendfontsize = 10,
             left_margin = 5 * Plots.mm, bottom_margin = 5 * Plots.mm,
             right_margin = 5 * Plots.mm, kwargs...)

    function curves!(plt, field; floored = false, labels)
        for id in ids
            sweep = sort(sweeps[id], by = r -> r.T)
            st = SWEEP_TEST_STYLE[id]
            y = [getproperty(r, field) for r in sweep]
            floored && (y = max.(y, OVERSHOOT_FLOOR))
            plot!(plt, [r.T for r in sweep], y; marker = st.mark, markersize = 3.5,
                  linewidth = 1.6, color = st.color, label = labels ? st.label : "")
        end
        return plt
    end

    function verticals!(plt; labels)
        vline!(plt, [T_PAPER]; color = :gray40, linestyle = :dash, linewidth = 1.5,
               label = labels ? latexstring("\\mathbb{T}($POLYDEG)") *
                                @sprintf(" = %.2e", T_PAPER) : "")
        vline!(plt, [T_used]; color = "#2F9E63", linewidth = 2.5,
               label = labels ? L"\mathbb{T}" * @sprintf(" selected = %.2e", T_used) : "")
    end

    left = panel(ylabel = L"L^2" * " error in " * L"A", legend = :bottomleft)
    curves!(left, :l2_A; labels = true)
    verticals!(left; labels = false)

    # Legend anchored by its bottom-left corner (axes fractions), clear of Test 3's
    # curve on the floor.
    right = panel(ylabel = "shock overshoot", legend = (0.637, 0.18), yticks = exp10.(-8:2:0))
    curves!(right, :overshoot; floored = true, labels = false)
    hline!(right, [tol]; color = "#2AA7C4", linestyle = :dashdot, linewidth = 1.4,
           label = @sprintf("overshoot cap = %.0e", tol))
    verticals!(right; labels = true)

    thesis_savefig(plot(left, right, layout = (1, 2), size = (1300, 560)), filename)
end

# ---------------------------------------------------------------------------------
# Threshold sweep and selection
# ---------------------------------------------------------------------------------
const RULE = "="^96

# Why T is inadmissible on this test ("" if admissible); `r` is a sweep row or
# `nothing` for a diverged run.
function gate_verdict(test, r; tol = OVERSHOOT_TOL)
    r === nothing && return "diverged"
    r.pattern_ok || return "pattern mismatch"
    r.overshoot > tol && return "oscillatory"
    (occursin('S', test.pattern) && r.frac_active <= 0) && return "indicator dormant"
    return ""
end

# Runs one test over T_SWEEP. Returns the completed runs and one table row per
# grid point (diverged runs included).
function sweep_grid(test, cfg)
    println("\n$RULE")
    println(" Test $(test.id) ($(test.pattern))  --  T sweep   " *
            "[L = $(cfg.L) cm, x0 = $(cfg.x0) cm, Tout = $(test.Tout) s, " *
            "alpha_max = $ALPHA_MAX, CFL = $(cfl_for(cfg.rsol))]")
    println(RULE)

    sweep = NamedTuple[]
    rows = NamedTuple[]
    for T in T_SWEEP
        note = isapprox(T, T_PAPER; rtol = 1e-6) ? "Hennemann et al. T($POLYDEG)" : ""
        r = try_run_case(test, cfg, T)
        if r === nothing
            @printf(" %-12s diverged\n", @sprintf("%.3e", T))
            push!(rows, (T = T, l1_A = NaN, l2_A = NaN, linf_A = NaN, l2_u = NaN,
                         pattern = "—", status = "diverged",
                         overshoot = NaN, fan_excursion = NaN,
                         tv_excess = NaN, frac_active = NaN, note = note))
            continue
        end
        pattern = classify_numerical(cfg, r)
        entry = merge(r, (; pattern, pattern_ok = pattern == test.pattern))
        push!(sweep, entry)
        verdict = gate_verdict(test, entry)
        @printf(" %-12s L2(A) %.4e  overshoot %.2e  %s\n", @sprintf("%.3e", T), r.l2_A,
                r.overshoot, isempty(verdict) ? "admissible" : verdict)
        push!(rows, (T = T, l1_A = r.l1_A, l2_A = r.l2_A, linf_A = r.linf_A,
                     l2_u = r.l2_u, pattern = pattern,
                     status = isempty(verdict) ? "admissible" : verdict,
                     overshoot = r.overshoot, fan_excursion = r.fan_excursion,
                     tv_excess = r.tv_excess, frac_active = r.frac_active, note = note))
    end
    return sweep, rows
end

# Selection rule: T is admissible if on every test the run completes, reproduces
# Toro's wave pattern, keeps the overshoot within OVERSHOOT_TOL, and (on tests
# with a shock) flags at least one element. Among admissible T, minimize
# max_tests L2(A; T) / min_admissible L2(A). Since L2(A) falls with T, this is
# the largest admissible T.
function select_T(sweeps; tol = OVERSHOOT_TOL)
    ids = sort(collect(keys(sweeps)))
    at(id, T) = (i = findfirst(r -> isapprox(r.T, T; rtol = 1e-6), sweeps[id]);
                 i === nothing ? nothing : sweeps[id][i])
    tests_by_id = Dict(t.id => t for t in TORO_TESTS)
    verdict(id, T) = gate_verdict(tests_by_id[id], at(id, T); tol = tol)

    admissible = filter(T -> all(id -> isempty(verdict(id, T)), ids), T_SWEEP)
    best = Dict(id => minimum(at(id, T).l2_A for T in admissible) for id in ids)
    ratio_keys = Tuple(Symbol("ratio_$id") for id in ids)

    rows = NamedTuple[]
    for T in T_SWEEP
        ratios = [(r = at(id, T); r === nothing ? NaN : r.l2_A / best[id]) for id in ids]
        adm = T in admissible
        failed = filter(p -> !isempty(p[2]), [(id, verdict(id, T)) for id in ids])
        overshoots = [(r = at(id, T); r === nothing ? NaN : r.overshoot) for id in ids]
        push!(rows, merge((T = T, admissible = adm ? "yes" : "no",
                           reason = adm ? "" : "$(failed[1][2]) (test $(failed[1][1]))",
                           ovr_max = all(isnan, overshoots) ? NaN :
                                     maximum(filter(!isnan, overshoots)),
                           score = adm ? maximum(ratios) : NaN),
                          NamedTuple{ratio_keys}(Tuple(ratios))))
    end

    scored = filter(r -> r.admissible == "yes", rows)
    return scored[argmin([r.score for r in scored])].T, rows
end

function sweep_table(test, cfg, rows, T_used)
    marked = [merge(r, (note = isapprox(r.T, T_used; rtol = 1e-6) ?
                               (isempty(r.note) ? "selected" : r.note * "; selected") :
                               r.note,))
              for r in rows]
    return TableSpec(
        "Test $(test.id) ($(test.pattern)) -- error and overshoot vs threshold T",
        [TableColumn(:T, "T"; fmt = :sci, digits = 4),
         TableColumn(:status, "status"; fmt = :str),
         TableColumn(:pattern, "pattern"; fmt = :str),
         TableColumn(:l1_A, "L1(A)"; fmt = :sci, digits = 4),
         TableColumn(:l2_A, "L2(A)"; fmt = :sci, digits = 4),
         TableColumn(:linf_A, "L∞(A)"; fmt = :sci, digits = 4),
         TableColumn(:l2_u, "L2(u)"; fmt = :sci, digits = 4),
         TableColumn(:overshoot, "shock ovr"; fmt = :sci, digits = 3),
         TableColumn(:fan_excursion, "fan stair"; fmt = :sci, digits = 3),
         TableColumn(:tv_excess, "TV exc"; fmt = :pctsigned, digits = 1),
         TableColumn(:frac_active, "α active"; fmt = :pct, digits = 1),
         TableColumn(:note, "note"; fmt = :str)],
        marked;
        notes = ["alpha_max = $ALPHA_MAX, " *
                 "CFL = $(cfl_for(cfg.rsol)), L = $(cfg.L) cm, " *
                 "Tout = $(test.Tout) s, $(2^REFINEMENT_LEVEL) cells, N = $POLYDEG.",
                 "'status' is this test's own verdict: 'admissible' = ran to " *
                 "completion, reproduced the $(test.pattern) pattern, kept the " *
                 "overshoot within the cap and (where the pattern has a shock) " *
                 "flagged at least one element; 'diverged' = area went negative " *
                 "and flux_ec raised a DomainError. The threshold has to clear it " *
                 "on all four tests at once (see the selection table).",
                 "The errors fall and the overshoot rises as T grows: no single T " *
                 "both minimizes the error and stays monotone at the shocks. The " *
                 "rule minimizes the error subject to the overshoot cap.",
                 "'shock ovr' is capped at $(@sprintf("%.0e", OVERSHOOT_TOL)) -- " *
                 "1 % of the exact A range -- which is the constraint that " *
                 "decides T on this test suite."])
end

function selection_table(rows, ids, T_used)
    marked = [merge(r, (; selected = isapprox(r.T, T_used; rtol = 1e-6) ? "<--" : ""))
              for r in rows]
    cols = TableColumn[TableColumn(:T, "T"; fmt = :sci, digits = 4),
                       TableColumn(:admissible, "admissible"; fmt = :str),
                       TableColumn(:reason, "why not"; fmt = :str),
                       TableColumn(:ovr_max, "worst ovr"; fmt = :sci, digits = 3)]
    for id in ids
        push!(cols, TableColumn(Symbol("ratio_$id"), "test $id"; fmt = :fixed, digits = 3))
    end
    push!(cols, TableColumn(:score, "worst"; fmt = :fixed, digits = 3))
    push!(cols, TableColumn(:selected, ""; fmt = :str))
    return TableSpec(
        "Threshold selection -- L2(A) relative to each test's best admissible value",
        cols, marked;
        notes = ["A T is admissible when EVERY test, at that T, (i) runs to " *
                 "completion and reproduces its Toro wave pattern, (ii) keeps its " *
                 "overshoot at or below $(@sprintf("%.0e", OVERSHOOT_TOL)) of the " *
                 "exact A range, and (iii) -- on the tests whose pattern contains " *
                 "a shock -- flags at least one element with the HG indicator.",
                 "'worst ovr' is the largest overshoot across the four tests at " *
                 "this T; it is the quantity gate (ii) is applied to.",
                 "'test i' = L2(A) at this T divided by the smallest L2(A) test i " *
                 "attains at any admissible T; 'worst' is the largest of those.",
                 "The selected T minimizes 'worst' over the admissible rows. L2 " *
                 "falls monotonically with T, so in practice the choice is the " *
                 "largest T the gate admits: as little blending as the overshoot " *
                 "cap allows."])
end

# ---------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------
function main(; run_sweep = false)
    @printf(" kappa = %.4f (CGS), c(A0) = %.2f cm/s, N = %d, %d cells\n",
            KAPPA, c_wave(A0, KAPPA), POLYDEG, 2^REFINEMENT_LEVEL)
    cfgs = Dict{Int, Any}()
    for test in TORO_TESTS
        cfg = configure_test(test)
        @assert wave_pattern(cfg.rsol) == test.pattern
        cfgs[test.id] = cfg
        @printf(" test %d: %-11s L = %.0f cm, max|u|/c = %.3f, CFL = %.2f\n", test.id,
                test.pattern, cfg.L, max_froude(cfg.rsol), cfl_for(cfg.rsol))
    end

    if run_sweep
        sweeps = Dict{Int, Any}()
        sweep_rows = Dict{Int, Any}()
        for test in TORO_TESTS
            sweeps[test.id], sweep_rows[test.id] = sweep_grid(test, cfgs[test.id])
        end
        T_used, sel_rows = select_T(sweeps)
        T_used == T_SELECTED ||
            error("the sweep selects T = $T_used, not T_SELECTED = $T_SELECTED")
        sel_spec = selection_table(sel_rows, sort(collect(keys(sweeps))), T_used)
        print_ascii(sel_spec)

        thesis_plot_style!()
        sweep_specs = TableSpec[]
        for test in TORO_TESTS
            # 337.5° lies between :right and :bottomright: Test 3's overshoot curve
            # runs along the bottom of the axes.
            sweep_legend = test.id == 3 ? 337.5 : :bottomright
            plot_T_sweep(sweeps[test.id], out("test$(test.id)_T_sweep.png"), T_used;
                         tol = OVERSHOOT_TOL, legend_pos = sweep_legend)
            push!(sweep_specs, sweep_table(test, cfgs[test.id], sweep_rows[test.id], T_used))
        end
        plot_T_sweep_combined(sweeps, out("T_sweep_all.png"), T_used; tol = OVERSHOOT_TOL)
    else
        T_used = T_SELECTED
    end

    # Production runs, with finer indicator sampling than the sweep.
    results = Dict{Int, Any}()
    for test in TORO_TESTS
        cfg = cfgs[test.id]
        println("\n Production run: test $(test.id) at T = $(@sprintf("%.3e", T_used))")
        r = run_case_quiet(test, cfg, T_used; entropy_check = true, indicator_interval = 20)
        results[test.id] = r

        thesis_plot_style!()
        area_legend = test.id == 3 ? :top : test.id == 4 ? :bottom : :best
        plot_profile(test, cfg, r, :A, out("test$(test.id)_area.png"); legend_pos = area_legend)
        plot_profile(test, cfg, r, :u, out("test$(test.id)_velocity.png"))
        plot_profile_error(test, cfg, r, :A, out("test$(test.id)_area_error.png"))
        plot_profile_error(test, cfg, r, :u, out("test$(test.id)_velocity_error.png"))
        plot_xt_diagram(test, cfg, out("test$(test.id)_xt_diagram.png"))
        thesis_plot_style!()
        thesis_savefig(plot_shock_indicator(r.shock_cb), out("test$(test.id)_shock_indicator.png"))
    end

    summary_rows = NamedTuple[]
    for test in TORO_TESTS
        r = results[test.id]
        push!(summary_rows,
              (test = test.id, exact = test.pattern,
               numerical = classify_numerical(cfgs[test.id], r),
               T = r.T, alpha_max = r.alpha_max,
               l1_A = r.l1_A, l2_A = r.l2_A, linf_A = r.linf_A,
               l1_u = r.l1_u, l2_u = r.l2_u, linf_u = r.linf_u,
               l2_Q = r.l2_Q, linf_Q = r.linf_Q,
               overshoot = r.overshoot, fan_excursion = r.fan_excursion,
               tv_excess = r.tv_excess, frac_active = r.frac_active,
               max_production = r.max_production))
    end

    err_spec = TableSpec(
        "Errors vs the exact solution, all tests at the selected T = " *
        "$(@sprintf("%.2e", T_used))",
        [TableColumn(:test, "test"; fmt = :int),
         TableColumn(:exact, "exact"; fmt = :str),
         TableColumn(:numerical, "numerical"; fmt = :str),
         TableColumn(:T, "T"; fmt = :sci, digits = 2),
         TableColumn(:alpha_max, "α_max"; fmt = :fixed, digits = 2),
         TableColumn(:l1_A, "L1(A)"; fmt = :sci, digits = 4),
         TableColumn(:l2_A, "L2(A)"; fmt = :sci, digits = 4),
         TableColumn(:linf_A, "L∞(A)"; fmt = :sci, digits = 4),
         TableColumn(:l1_u, "L1(u)"; fmt = :sci, digits = 4),
         TableColumn(:l2_u, "L2(u)"; fmt = :sci, digits = 4),
         TableColumn(:linf_u, "L∞(u)"; fmt = :sci, digits = 4)],
        summary_rows;
        notes = ["Errors are volume-normalized, on the AnalysisCallback's " *
                 "over-integrated quadrature nodes, against the closed-form " *
                 "solution at Tout.",
                 "'exact' vs 'numerical' compares Toro's Table 9.2 wave pattern " *
                 "with the one the discrete solution actually produced."])
    cons_spec = TableSpec(
        "Conservative-variable errors and shock-capturing diagnostics",
        [TableColumn(:test, "test"; fmt = :int),
         TableColumn(:l2_Q, "L2(Q)"; fmt = :sci, digits = 4),
         TableColumn(:linf_Q, "L∞(Q)"; fmt = :sci, digits = 4),
         TableColumn(:overshoot, "shock ovr"; fmt = :sci, digits = 3),
         TableColumn(:fan_excursion, "fan stair"; fmt = :sci, digits = 3),
         TableColumn(:tv_excess, "TV exc"; fmt = :pctsigned, digits = 1),
         TableColumn(:frac_active, "α active"; fmt = :pct, digits = 1),
         TableColumn(:max_production, "max Π"; fmt = :sci, digits = 4)],
        summary_rows;
        notes = ["Π = numerical entropy production; must stay ≤ 0 (dissipative).",
                 "'α active' = peak fraction of elements the HG indicator flagged.",
                 "'shock ovr' is the overshoot as a fraction of the exact A " *
                 "range. The selected T is the largest one holding it at or " *
                 "below $(@sprintf("%.0e", OVERSHOOT_TOL)) on every test -- the " *
                 "constraint the error minimization runs under, not a free " *
                 "diagnostic. See the selection table in riemann_T_sweep.md."])

    print_ascii(err_spec)
    print_ascii(cons_spec)
    emit_tables([err_spec, cons_spec], OUTDIR, "riemann_errors"; print = false,
                title = "Riemann problem -- errors and diagnostics " *
                        "(Toro 2025, Ch. 9, Table 9.2)")
    run_sweep &&
        emit_tables(vcat([sel_spec], sweep_specs), OUTDIR, "riemann_T_sweep"; print = false,
                    title = "Riemann problem -- threshold sweep and selection " *
                            "(Toro 2025, Ch. 9, Table 9.2)")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(run_sweep = "--T-sweep" in ARGS)
end
