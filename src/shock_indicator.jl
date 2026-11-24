# Hennemann-Gassner shock indicator with a free threshold T, and a callback that
# records its blending factor. Trixi names the blending factor `alpha`; the thesis
# calls it θ, since α there is the momentum-flux correction coefficient.
#
# The call method below is adapted from Trixi.jl's `IndicatorHennemannGassner`
# (MIT License, Copyright (c) 2020-present The Trixi.jl Authors); only the
# threshold line differs.

"""
    hg_paper_threshold(basis)

Threshold of Hennemann et al. (2021), eq. (42), `T(N) = 0.5 · 10^(-1.8 (N+1)^{1/4})`,
written exactly as in Trixi so that it reproduces `IndicatorHennemannGassner` bit
for bit.
"""
@inline function hg_paper_threshold(basis::LobattoLegendreBasis)
    RealT = real(basis)
    return 0.5f0 * 10^(convert(RealT, -1.8) * nnodes(basis)^convert(RealT, 0.25))
end

"""
    IndicatorHennemannGassnerT(equations, basis; alpha_max = 0.5, alpha_min = 0.001,
                               alpha_smooth = true, variable, T = hg_paper_threshold(basis))

`IndicatorHennemannGassner` with the sigmoid threshold `T` as a parameter. `T` is
the centre of the sigmoid `θ = 1/(1 + exp(-s/T (E - T)))`, so a smaller `T` blends
in more finite-volume dissipation.
"""
struct IndicatorHennemannGassnerT{RealT <: Real, Variable, Cache} <: AbstractIndicator
    alpha_max::RealT
    alpha_min::RealT
    alpha_smooth::Bool
    variable::Variable
    cache::Cache
    T::RealT
end

function IndicatorHennemannGassnerT(equations::AbstractEquations, basis;
                                    alpha_max = 0.5, alpha_min = 0.001,
                                    alpha_smooth = true, variable,
                                    T = hg_paper_threshold(basis))
    cache = Trixi.create_cache(IndicatorHennemannGassner, equations, basis)
    alpha_max, alpha_min, T = promote(alpha_max, alpha_min, T)
    return IndicatorHennemannGassnerT{typeof(alpha_max), typeof(variable), typeof(cache)}(
        alpha_max, alpha_min, alpha_smooth, variable, cache, T)
end

function (indicator::IndicatorHennemannGassnerT)(u, mesh, equations, dg::DGSEM, cache;
                                                 kwargs...)
    (; alpha_smooth) = indicator
    (; alpha, alpha_tmp) = indicator.cache
    resize!(alpha, nelements(dg, cache))
    alpha_smooth && resize!(alpha_tmp, nelements(dg, cache))

    RealT = real(dg)
    threshold = convert(RealT, indicator.T)
    o_0001 = convert(RealT, 0.0001)
    parameter_s = log((1 - o_0001) / o_0001)

    Trixi.@threaded for element in eachelement(dg, cache)
        Trixi.calc_indicator_hennemann_gassner!(indicator, threshold, parameter_s, u,
                                                element, mesh, equations, dg, cache)
    end

    alpha_smooth && Trixi.apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
    return alpha
end

# Records mean and max of the per-element blending factor and the number of
# active elements (θ > alpha_min). It reads the indicator's cached θ, which
# reflects the last right-hand-side evaluation before the callback fires: the
# last RK stage, or any later rhs! call made by a callback earlier in the set.
mutable struct ShockIndicatorMonitor{Semi, RealT}
    semi::Semi
    alpha_max::RealT
    alpha_min::RealT
    n_elements::Int
    t::Vector{RealT}
    alpha_mean::Vector{RealT}
    alpha_max_val::Vector{RealT}
    n_active::Vector{Int}
end

function (monitor::ShockIndicatorMonitor)(integrator)
    _, _, dg, _ = Trixi.mesh_equations_solver_cache(monitor.semi)
    alpha = dg.volume_integral.indicator.cache.alpha
    push!(monitor.t, integrator.t)
    push!(monitor.alpha_mean, mean(alpha))
    push!(monitor.alpha_max_val, maximum(alpha))
    push!(monitor.n_active, count(>(monitor.alpha_min), alpha))
    u_modified!(integrator, false)
    return nothing
end

"""
    ShockIndicatorMonitorCallback(semi; interval = 100)

Records the blending factor every `interval` accepted steps and at the start.
Retrieve the data with `shock_indicator_history(cb)`.
"""
function ShockIndicatorMonitorCallback(semi; interval = 100)
    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    indicator = dg.volume_integral.indicator
    RealT = typeof(indicator.alpha_max)
    monitor = ShockIndicatorMonitor(semi, indicator.alpha_max, indicator.alpha_min,
                                    nelements(dg, cache), RealT[], RealT[], RealT[], Int[])
    condition(u, t, integrator) = integrator.stats.naccept % interval == 0
    return DiscreteCallback(condition, monitor;
                            initialize = (cb, u, t, integrator) -> cb.affect!(integrator),
                            save_positions = (false, false))
end

shock_indicator_history(cb::DiscreteCallback) = cb.affect!

# Legend anchors (axes fractions of the box's top-left corner, GR backend): just
# below where :topright would put them, so the key clears the max-θ trace.
const SI_LEGEND_Y = 0.87
const SI_LEGEND_X_THETA = 0.838
const SI_LEGEND_X_COUNT = 0.808

# Whole-number ticks for an element count, from a nice-number ladder.
function count_ticks(nmax::Int; target = 5)
    nmax <= 0 && return 0:1:1
    step = 1
    for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)
        step = s
        nmax / s <= target && break
    end
    return 0:step:(cld(nmax, step) * step)
end

"""
    plot_shock_indicator(cb; size = (960, 760))

Mean and max θ over time with the θ_max ceiling (top), and the number of active
elements (bottom).
"""
function plot_shock_indicator(cb::DiscreteCallback; size = (960, 760))
    d = cb.affect!
    t = d.t
    xl = isempty(t) ? (0, 1) : extrema(t)

    p1 = plot(t, d.alpha_mean; label = "mean " * L"\theta", color = :steelblue, lw = 2)
    plot!(p1, t, d.alpha_max_val; label = "max " * L"\theta", color = :firebrick, lw = 2)
    hline!(p1, [d.alpha_max]; color = :gray, ls = :dash,
           label = L"\theta_{\max}" * " (ceiling)")
    plot!(p1; ylabel = L"\theta", xlims = xl,
          guidefontsize = GUIDEFONT, tickfontsize = TICKFONT, legendfontsize = LEGENDFONT,
          legend = (SI_LEGEND_X_THETA, SI_LEGEND_Y), framestyle = :box, gridalpha = 0.25)

    p2 = plot(t, d.n_active; label = "active elements", color = :seagreen, lw = 2,
              fillrange = 0, fillalpha = 0.15, fillcolor = :seagreen)
    nmax = isempty(d.n_active) ? 0 : maximum(d.n_active)
    ytop = max(nmax * 1.08, 1.0)
    n_ticks = filter(<=(ytop), collect(count_ticks(nmax)))
    plot!(p2; xlabel = L"t", ylabel = "# elements (out of $(d.n_elements))",
          xlims = xl, yticks = n_ticks, ylims = (0, ytop),
          guidefontsize = GUIDEFONT, tickfontsize = TICKFONT, legendfontsize = LEGENDFONT,
          legend = (SI_LEGEND_X_COUNT, SI_LEGEND_Y), framestyle = :box, gridalpha = 0.25)

    return plot(p1, p2; layout = (2, 1), size = size, link = :x)
end
