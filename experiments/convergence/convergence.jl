# Order of accuracy with a manufactured solution (thesis Section 4.1): h-convergence
# for N = 1..5 on 4..512 elements and p-convergence at a fixed element size.
# Writes results/convergence/{convergence_h_A, convergence_h_Q, p_convergence_errors,
# p_convergence_rates}.png and convergence_errors.md.
using Trixi
using OrdinaryDiffEqLowStorageRK
using Plots
using LaTeXStrings
using Printf
using BloodFlowDGSEM

thesis_plot_style!()

const OUTDIR = results_dir("convergence")
const ELIXIR = joinpath(@__DIR__, "elixir_convergence.jl")

const L = 20.0
const POLYDEGS = 1:5
const LEVELS = 2:9   # h = L / 2^level

# Time integrator and CFL per degree. The integrator order must be at least the
# spatial order N + 1, otherwise its error eventually dominates the h-curve. For
# N = 1, 2 the orders are equal (ORK256, RDPK3Sp35). For N >= 3 each CFL is the
# largest that reproduces time-converged errors to better than 0.1 % on every
# mesh above the round-off floor, backed off one notch.
const ALGS = Dict(:ork256 => () -> ORK256(),
                  :rdpk35 => () -> RDPK3Sp35(),
                  :ck54 => () -> CarpenterKennedy2N54(williamson_condition = false),
                  :rdpk510 => () -> RDPK3Sp510())
const ALG_NAMES = Dict(:ork256 => "ORK256", :rdpk35 => "RDPK3Sp35",
                       :ck54 => "CarpenterKennedy2N54", :rdpk510 => "RDPK3Sp510")
const SWEEP = Dict(1 => (alg = :ork256, cfl = 1.25),
                   2 => (alg = :rdpk35, cfl = 1.0),
                   3 => (alg = :ck54, cfl = 0.75),
                   4 => (alg = :rdpk510, cfl = 1.0),
                   5 => (alg = :rdpk510, cfl = 0.5))

h(level) = L / 2.0^level
hs = [h(level) for level in LEVELS]
const NLEVELS = length(LEVELS)

results = Dict{Int, Any}()
for polydeg in POLYDEGS
    s = SWEEP[polydeg]
    println("="^100)
    println("Running convergence test for polydeg = $polydeg ",
            "($(ALG_NAMES[s.alg]), CFL = $(s.cfl))")
    println("="^100)
    _, errorsmatrix = Trixi.convergence_test(ELIXIR, NLEVELS; polydeg = polydeg,
                                             initial_refinement_level = first(LEVELS),
                                             cfl = s.cfl, time_integrator = ALGS[s.alg]())
    results[polydeg] = errorsmatrix
end

l2(polydeg, var) = results[polydeg][:l2][:, var]   # var 1 = A, 2 = Q

# Round-off saturation. The A error bottoms out at about 1e-14, so a mesh counts
# as round-off dominated if L2(A) is below ROUNDOFF_FLOOR_A and the EOC of A or Q
# has left N + 1 by more than EOC_TOL; every finer mesh is then flagged too. The
# test uses A only: the exact Q is zero and its error is driven by A's, and once
# A saturates Q's floor grows like 1/h.
const ROUNDOFF_FLOOR_A = 1e-13
const EOC_TOL = 0.25

observed_orders(errs) = [i == 1 ? NaN : log2(errs[i - 1] / errs[i]) for i in eachindex(errs)]

function saturation_mask(polydeg)
    off_order(eoc) = abs.(eoc .- (polydeg + 1)) .> EOC_TOL
    off = off_order(observed_orders(l2(polydeg, 1))) .|
          off_order(observed_orders(l2(polydeg, 2)))
    return accumulate(|, (l2(polydeg, 1) .< ROUNDOFF_FLOOR_A) .& off)
end
saturated = Dict(polydeg => saturation_mask(polydeg) for polydeg in POLYDEGS)

# Tick labels 10⁻ⁿ in the axis font (LaTeX strings would switch to the math font).
const SUPERSCRIPT_DIGITS = Dict(zip("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
pow10_label(k::Integer) = "10" * map(c -> SUPERSCRIPT_DIGITS[c], string(k))

# Decade limits spanning `values`, widened to whole `step`-decade intervals.
function decade_axis(values, step)
    e_hi = ceil(Int, log10(maximum(values)))
    e_lo = e_hi - step * cld(e_hi - floor(Int, log10(minimum(values))), step)
    exps = e_hi:-step:e_lo
    return (10.0^e_lo, 10.0^e_hi), (10.0 .^ exps, pow10_label.(exps))
end

# Round-off points are drawn as open circles; GR cannot draw a hollow :circle.
const MARKER_HOLLOW_CIRCLE = Shape(cos.(range(0, 2π; length = 72)),
                                   sin.(range(0, 2π; length = 72)))
const HOLLOW = (markershape = MARKER_HOLLOW_CIRCLE, markersize = 6,
                markercolor = RGBA(1, 1, 1, 0), markerstrokealpha = 1.0,
                markerstrokewidth = 1.3)

# ---------------------------------------------------------------------------------
# h-convergence figure: one file per variable, common y-axis
# ---------------------------------------------------------------------------------
all_errors = [e for polydeg in POLYDEGS for var in 1:2 for e in l2(polydeg, var)]
ylims_h, yticks_h = decade_axis(all_errors, 4)

const PANEL_SIZE = (1200, 480)
common_opts = (size = PANEL_SIZE, xscale = :log10, yscale = :log10,
               xlabel = "Element Size", ylabel = L"L^2" * " Error", legend = false,
               xlims = (0.03, 7), ylims = ylims_h, yticks = yticks_h,
               xticks = ([0.03, 0.1, 0.3, 1, 3], ["0.03", "0.1", "0.3", "1", "3"]),
               guidefontsize = 13, tickfontsize = 11, legendfontsize = 11,
               left_margin = 10Plots.mm, bottom_margin = 6Plots.mm,
               right_margin = 4Plots.mm)
plot_A = plot(; common_opts...)
plot_Q = plot(; common_opts...)

const DEGREE_COLORS = Dict(1 => "#2F6FC0", 2 => "#D6402E", 3 => "#2F9E63",
                           4 => "#E5732A", 5 => "#6E4CC4")

for polydeg in POLYDEGS, (panel, var) in ((plot_A, 1), (plot_Q, 2))
    errors = l2(polydeg, var)
    color = DEGREE_COLORS[polydeg]
    order = polydeg + 1
    k = something(findlast(!, saturated[polydeg]), 0)   # clean points are 1:k

    plot!(panel, hs[1:k], errors[1:k], marker = :circle, color = color,
          label = "degree $polydeg")
    if k < NLEVELS
        plot!(panel, hs[max(k, 1):end], errors[max(k, 1):end], color = color, label = "")
        scatter!(panel, hs[(k + 1):end], errors[(k + 1):end]; HOLLOW...,
                 markerstrokecolor = color, label = "")
    end

    # Reference slope anchored at the finest clean point.
    k >= 2 || continue
    ref = errors[k] .* (hs[1:k] ./ hs[k]) .^ order
    plot!(panel, hs[1:k], ref, linestyle = :dash, color = color,
          label = "order $order reference")
end

# Legend strip under the Q panel; the Q file is taller by its height.
const LEGEND_HEIGHT = 60
leg_h = plot(; framestyle = :none, legend = (0.0, 1.0), legend_columns = 5,
             legendfontsize = 11, foreground_color_legend = nothing,
             left_margin = 4Plots.mm, top_margin = 0Plots.mm, bottom_margin = 0Plots.mm)
for polydeg in POLYDEGS
    plot!(leg_h, [NaN], [NaN], marker = :circle, color = DEGREE_COLORS[polydeg],
          label = "degree $polydeg")
end
for polydeg in POLYDEGS
    plot!(leg_h, [NaN], [NaN], linestyle = :dash, color = DEGREE_COLORS[polydeg],
          label = "order $(polydeg + 1) reference")
end

h_total = PANEL_SIZE[2] + LEGEND_HEIGHT
fig_Q = plot(plot_Q, leg_h,
             layout = grid(2, 1, heights = [PANEL_SIZE[2], LEGEND_HEIGHT] ./ h_total),
             size = (PANEL_SIZE[1], h_total))
thesis_savefig(plot_A, joinpath(OUTDIR, "convergence_h_A.png"))
thesis_savefig(fig_Q, joinpath(OUTDIR, "convergence_h_Q.png"))

# ---------------------------------------------------------------------------------
# p-convergence: at fixed element size, fit e_N ≈ c exp(-bN) over the clean
# degrees, and the local rates b_N = ln(e_N / e_{N+1}). Since e_N ∝ Δz^{N+1}
# asymptotically, each halving of Δz should raise every b_N by ln 2.
# ---------------------------------------------------------------------------------

# Least squares for ln e = ln c - bN; R² needs at least three points.
function fit_exponential(Ns, errs)
    length(Ns) >= 3 || return (sigma = NaN, C = NaN, r2 = NaN, npts = length(Ns))
    x = Float64.(Ns)
    y = log.(errs)
    xm, ym = sum(x) / length(x), sum(y) / length(y)
    slope = sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
    intercept = ym - slope * xm
    r2 = 1 - sum((y .- (intercept .+ slope .* x)) .^ 2) / sum((y .- ym) .^ 2)
    return (sigma = -slope, C = exp(intercept), r2 = r2, npts = length(Ns))
end

clean_degrees(i) = [polydeg for polydeg in POLYDEGS if !saturated[polydeg][i]]

local_rate(i, var, polydeg) =
    saturated[polydeg][i] || saturated[polydeg + 1][i] ? NaN :
    log(l2(polydeg, var)[i] / l2(polydeg + 1, var)[i])

const JUMPS = first(POLYDEGS):(last(POLYDEGS) - 1)

fits = [(A = fit_exponential(clean_degrees(i), [l2(N, 1)[i] for N in clean_degrees(i)]),
         Q = fit_exponential(clean_degrees(i), [l2(N, 2)[i] for N in clean_degrees(i)]))
        for i in 1:NLEVELS]
rates = Dict(var => [local_rate(i, var, N) for i in 1:NLEVELS, N in JUMPS] for var in 1:2)

# The finest mesh on which every degree is still clean.
istar = findlast(i -> length(clean_degrees(i)) == length(POLYDEGS), 1:NLEVELS)
hstar = hs[istar]
println("p-convergence evaluated at level $(LEVELS[istar]) (Δz = $hstar, ",
        Int(L / hstar), " cells)")

const VAR_COLORS = ("#2F6FC0", "#D6402E")
const VAR_NAMES = (L"A", L"Q")
Ns = collect(POLYDEGS)

pstar_errors = [l2(N, var)[istar] for N in POLYDEGS for var in 1:2]
ylims_p, yticks_p = decade_axis(pstar_errors, 2)

plot_err = plot(; size = PANEL_SIZE, yscale = :log10, xlabel = "Polynomial Degree " * L"N",
                ylabel = L"L^2" * " Error", xticks = Ns, xlims = (0.6, last(Ns) + 0.4),
                ylims = ylims_p, yticks = yticks_p, legend = :topright,
                guidefontsize = 13, tickfontsize = 11, legendfontsize = 11,
                left_margin = 10Plots.mm, bottom_margin = 6Plots.mm,
                right_margin = 4Plots.mm)
for var in 1:2
    fit = fits[istar][var]
    errors = [l2(N, var)[istar] for N in POLYDEGS]
    scatter!(plot_err, Ns, errors, marker = :circle, color = VAR_COLORS[var],
             label = VAR_NAMES[var])
    plot!(plot_err, Ns, fit.C .* exp.(-fit.sigma .* Ns), linestyle = :dash,
          color = VAR_COLORS[var], label = "fit, " * L"b" * @sprintf(" = %.2f", fit.sigma))
end

# Local rates of A at every level with at least two clean jumps, light to dark.
rate_levels = [i for i in 1:NLEVELS if count(!isnan, rates[1][i, :]) >= 2]
level_colors = palette(["#9DB9E0", "#1B3F73"], max(length(rate_levels), 2))
plot_rate = plot(; size = PANEL_SIZE, xlabel = "Degree Jump " * L"N \to N+1",
                 ylabel = "Local Rate " * L"b_N" * " (" * L"A" * ")",
                 xticks = (collect(JUMPS), ["$N→$(N + 1)" for N in JUMPS]),
                 xlims = (first(JUMPS) - 0.4, last(JUMPS) + 0.4), legend = :topright,
                 guidefontsize = 13, tickfontsize = 11, legendfontsize = 11,
                 left_margin = 10Plots.mm, bottom_margin = 6Plots.mm,
                 right_margin = 4Plots.mm)
for (j, i) in enumerate(rate_levels)
    plot!(plot_rate, collect(JUMPS), rates[1][i, :], marker = :circle,
          color = level_colors[j], label = L"\Delta z" * @sprintf(" = %g", hs[i]))
end
hline!(plot_rate, [fits[istar].A.sigma], linestyle = :dash, color = :gray40,
       label = "fit, " * L"\Delta z" * @sprintf(" = %g", hstar))

thesis_savefig(plot_err, joinpath(OUTDIR, "p_convergence_errors.png"))
thesis_savefig(plot_rate, joinpath(OUTDIR, "p_convergence_rates.png"))

# ---------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------
roundoff_note = "Rows marked \"round-off\" have L2(A) < $ROUNDOFF_FLOOR_A and an L2 " *
                "EOC of A or Q more than $EOC_TOL from the expected order (and every " *
                "finer row is marked too); Q is flagged with A because its error is " *
                "driven by A's."

specs = TableSpec[]
for polydeg in POLYDEGS, norm in (:l2, :linf)
    errors_A = results[polydeg][norm][:, 1]
    errors_Q = results[polydeg][norm][:, 2]
    eoc_A = observed_orders(errors_A)
    eoc_Q = observed_orders(errors_Q)
    sat = saturated[polydeg]

    rows = NamedTuple[(cells = Int(L / hs[i]), h = hs[i],
                       err_A = errors_A[i], eoc_A = eoc_A[i],
                       err_Q = errors_Q[i], eoc_Q = eoc_Q[i],
                       note = sat[i] ? "round-off" : nothing)
                      for i in 1:NLEVELS]
    # Mean over the (up to) three finest orders whose finer mesh is clean.
    clean_pairs = last([i for i in 2:NLEVELS if !sat[i]], 3)
    mean_over(eoc) = isempty(clean_pairs) ? NaN : sum(eoc[clean_pairs]) / length(clean_pairs)
    push!(rows, (cells = nothing, h = NaN, err_A = NaN, eoc_A = mean_over(eoc_A),
                 err_Q = NaN, eoc_Q = mean_over(eoc_Q), note = nothing))

    norm_label = norm === :l2 ? "L2" : "L∞"
    push!(specs, TableSpec(
        "Polynomial degree $polydeg, $norm_label error and observed order",
        [TableColumn(:cells, "cells"; fmt = :int),
         TableColumn(:h, "Δz"; fmt = :sci, digits = 3),
         TableColumn(:err_A, "$norm_label(A)"; fmt = :sci, digits = 3),
         TableColumn(:eoc_A, "EOC(A)"; fmt = :fixed, digits = 2),
         TableColumn(:err_Q, "$norm_label(Q)"; fmt = :sci, digits = 3),
         TableColumn(:eoc_Q, "EOC(Q)"; fmt = :fixed, digits = 2),
         TableColumn(:note, "note"; fmt = :str)],
        rows;
        notes = ["Expected order for degree $polydeg is $(polydeg + 1).",
                 "Time integrator: $(ALG_NAMES[SWEEP[polydeg].alg]) at " *
                 "CFL = $(SWEEP[polydeg].cfl).",
                 "Last row is the mean observed order over the (up to) three " *
                 "finest pairs whose finer mesh is not round-off dominated; " *
                 "blank cells have no predecessor to compare against.",
                 roundoff_note]))
end

push!(specs, TableSpec(
    "p-convergence at fixed Δz = $hstar ($(Int(L / hstar)) cells), L2 error vs degree",
    [TableColumn(:N, "N"; fmt = :int),
     TableColumn(:err_A, "L2(A)"; fmt = :sci, digits = 3),
     TableColumn(:sig_A, "b(N→N+1, A)"; fmt = :fixed, digits = 2),
     TableColumn(:err_Q, "L2(Q)"; fmt = :sci, digits = 3),
     TableColumn(:sig_Q, "b(N→N+1, Q)"; fmt = :fixed, digits = 2)],
    [(N = N, err_A = l2(N, 1)[istar], err_Q = l2(N, 2)[istar],
      sig_A = N in JUMPS ? rates[1][istar, N - first(JUMPS) + 1] : NaN,
      sig_Q = N in JUMPS ? rates[2][istar, N - first(JUMPS) + 1] : NaN)
     for N in POLYDEGS];
    notes = [@sprintf("Least-squares fit e_N ≈ c_fit exp(-bN): b(A) = %.3f (R² = %.4f), b(Q) = %.3f (R² = %.4f).",
                      fits[istar].A.sigma, fits[istar].A.r2,
                      fits[istar].Q.sigma, fits[istar].Q.r2),
             "Local rates b_N = ln(e_N / e_{N+1}); constant means exactly exponential, " *
             "increasing means faster than exponential.",
             "Δz is the finest mesh on which no degree is round-off dominated.",
             "Time integrators: " *
             join(["N = $polydeg: $(ALG_NAMES[SWEEP[polydeg].alg]) @ CFL " *
                   "$(SWEEP[polydeg].cfl)" for polydeg in POLYDEGS], "; ") * "."]))

push!(specs, TableSpec(
    "Exponential fit e_N ≈ c_fit exp(-bN) of the L2 error at each element size",
    [TableColumn(:cells, "cells"; fmt = :int),
     TableColumn(:h, "Δz"; fmt = :sci, digits = 3),
     TableColumn(:npts, "degrees fit"; fmt = :int),
     TableColumn(:sig_A, "b(A)"; fmt = :fixed, digits = 3),
     TableColumn(:r2_A, "R²(A)"; fmt = :fixed, digits = 4),
     TableColumn(:sig_Q, "b(Q)"; fmt = :fixed, digits = 3),
     TableColumn(:r2_Q, "R²(Q)"; fmt = :fixed, digits = 4)],
    [(cells = Int(L / hs[i]), h = hs[i], npts = fits[i].A.npts,
      sig_A = fits[i].A.sigma, r2_A = fits[i].A.r2,
      sig_Q = fits[i].Q.sigma, r2_Q = fits[i].Q.r2) for i in 1:NLEVELS];
    notes = ["Only degrees that are not round-off dominated are fit; fewer than " *
             "three leaves the fit blank.",
             "b fitted over different numbers of degrees is not directly comparable " *
             "across rows; the local-rate tables below are."]))

for (var, name) in ((1, "A"), (2, "Q"))
    R = rates[var]
    increments = [begin
                      d = [R[i, j] - R[i - 1, j] for i in 2:NLEVELS
                           if !isnan(R[i, j]) && !isnan(R[i - 1, j])]
                      isempty(d) ? NaN : sum(d) / length(d)
                  end for j in eachindex(JUMPS)]
    keys_ = [Symbol("s$(N)$(N + 1)") for N in JUMPS]
    rows = NamedTuple[merge((cells = Int(L / hs[i]), h = hs[i]),
                            NamedTuple{Tuple(keys_)}(Tuple(R[i, :])))
                      for i in 1:NLEVELS]
    push!(rows, merge((cells = nothing, h = NaN), NamedTuple{Tuple(keys_)}(Tuple(increments))))
    push!(specs, TableSpec(
        "Local p-rates b_N = ln(e_N / e_{N+1}) of L2($name) at each element size",
        vcat([TableColumn(:cells, "cells"; fmt = :int),
              TableColumn(:h, "Δz"; fmt = :sci, digits = 3)],
             [TableColumn(k, "$N→$(N + 1)"; fmt = :fixed, digits = 2)
              for (k, N) in zip(keys_, JUMPS)]),
        rows;
        notes = ["Last row is the mean increase of each rate per halving of Δz; " *
                 @sprintf("asymptotically it should be ln 2 = %.3f.", log(2)),
                 "Blank cells: at least one of the two degrees is round-off dominated."]))
end

emit_tables(specs, OUTDIR, "convergence_errors";
            title = "Convergence test -- errors and observed orders " *
                    "(manufactured solution, L = $L cm)")

println()
println("p-convergence summary (Δz = $hstar):")
@printf("  b(A) = %.3f  (R² = %.4f)\n", fits[istar].A.sigma, fits[istar].A.r2)
@printf("  b(Q) = %.3f  (R² = %.4f)\n", fits[istar].Q.sigma, fits[istar].Q.r2)
