# Coupled 1D-0D system for a Windkessel outlet. The capacitor pressure is appended
# to the DG state, y = [U_h; p_C] (plus the cumulative inlet and R2 volumes when
# `track_volume = true`), so the time integrator advances the 0D model with the
# DG solution and the coupling has the integrator's order.

struct SemidiscretizationWindkessel{Semi, BC, F}
    semi::Semi
    bc::BC          # the same object placed at x_pos in `semi`
    f!::F           # Trixi's semidiscrete right-hand side
    n::Int          # length of the DG part of y
    element_out::Int
    node_out::Int
    element_in::Int
    track_volume::Bool
end

dg_state(y, w::SemidiscretizationWindkessel) = view(y, 1:w.n)

function end_node(semi, side::Symbol)
    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    x = cache.elements.node_coordinates
    best = side === :max ? typemin(eltype(x)) : typemax(eltype(x))
    el, node = 1, 1
    for element in Trixi.eachelement(dg, cache), i in Trixi.eachnode(dg)
        if (side === :max && x[1, i, element] > best) ||
           (side === :min && x[1, i, element] < best)
            best = x[1, i, element]
            el, node = element, i
        end
    end
    return el, node
end

function outlet_trace(w::SemidiscretizationWindkessel, u_dg)
    _, equations, dg, _ = Trixi.mesh_equations_solver_cache(w.semi)
    u = Trixi.wrap_array(u_dg, w.semi)
    return Trixi.get_node_vars(u, equations, dg, w.node_out, w.element_out)
end

function windkessel_rhs!(dy, y, w::SemidiscretizationWindkessel, t)
    n = w.n
    bc = w.bc
    p_C = y[n + 1]
    bc.p_C = p_C
    w.f!(view(dy, 1:n), view(y, 1:n), w.semi, t)

    # Q_J is the outlet mass flux the DG evaluation just used. On a 1D TreeMesh
    # Trixi stores boundary fluxes as surface_flux_values[variable, direction,
    # element], direction 1 = x_neg, 2 = x_pos.
    _, _, _, cache = Trixi.mesh_equations_solver_cache(w.semi)
    boundary_fluxes = cache.elements.surface_flux_values
    Q_J = boundary_fluxes[1, 2, w.element_out]
    dy[n + 1] = (Q_J - (p_C - bc.p_out) / bc.R2) / bc.C

    if w.track_volume
        # The numerical inlet mass flux, not Q_in(t): the two differ by the
        # dissipation term of the surface flux.
        dy[n + 2] = boundary_fluxes[1, 1, w.element_in]
        dy[n + 3] = (p_C - bc.p_out) / bc.R2
    end
    return nothing
end

# StepsizeCallback: the CFL condition is that of the DG part.
function Trixi.calculate_dt(u_ode, t, cfl_hyperbolic, cfl_parabolic,
                            w::SemidiscretizationWindkessel)
    return Trixi.calculate_dt(view(u_ode, 1:w.n), t, cfl_hyperbolic, cfl_parabolic, w.semi)
end

"""
    semidiscretize_windkessel(semi, bc, tspan; track_volume = false)

`ODEProblem` for `y = [U_h; p_C]` with `p_C(0) = bc.p_C0`. `bc` must be the
Windkessel condition at `x_pos` of `semi` (a 1D `TreeMesh`). Trixi callbacks see
`ode.p`, a `SemidiscretizationWindkessel`; only `StepsizeCallback` is supported.
"""
function semidiscretize_windkessel(semi, bc::BoundaryConditionWindkessel3Element, tspan;
                                   track_volume::Bool = false)
    ode_dg = semidiscretize(semi, tspan)
    u0 = ode_dg.u0
    element_out, node_out = end_node(semi, :max)
    element_in, _ = end_node(semi, :min)
    w = SemidiscretizationWindkessel(semi, bc, ode_dg.f, length(u0), element_out, node_out,
                                     element_in, track_volume)
    bc.p_C = bc.p_C0
    y0 = track_volume ? [u0; bc.p_C0; zero(bc.p_C0); zero(bc.p_C0)] : [u0; bc.p_C0]
    return ODEProblem(windkessel_rhs!, y0, tspan, w)
end

"""
    windkessel_history(sol, w)

Outlet time series at the saved states: `t`, `p_C`, the junction pressure
`p_out = p(A_b)`, the current `Q_out = F̂₁(U_i, U_b)`, the exterior state
`A_out`, `u_out`, the interior trace `A_trace`, `u_trace`, and the junction
residual `resid_norm`.
"""
function windkessel_history(sol, w::SemidiscretizationWindkessel)
    _, equations, dg, _ = Trixi.mesh_equations_solver_cache(w.semi)
    surface_flux = dg.surface_integral.surface_flux
    bc = w.bc
    RealT = eltype(sol.u[1])
    cols = [RealT[] for _ in 1:8]
    p_C_current = bc.p_C
    for y in sol.u
        bc.p_C = y[w.n + 1]
        u_i = outlet_trace(w, dg_state(y, w))
        A_i, vel_i = cons2prim(u_i, equations)
        u_b = windkessel_exterior(bc, u_i, 1, surface_flux, equations)
        Q_J = surface_flux(u_i, u_b, 1, equations)[1]
        A_b, vel_b = cons2prim(u_b, equations)
        resid = abs(windkessel_residual(log(A_b),
                                        (u_i, bc.R1, bc.p_C, 1, surface_flux, equations)))
        for (c, v) in zip(cols, (bc.p_C, pressure(u_b, equations), Q_J, A_b, vel_b,
                                 A_i, vel_i, resid))
            push!(c, v)
        end
    end
    bc.p_C = p_C_current
    return (t = collect(sol.t), p_C = cols[1], p_out = cols[2], Q_out = cols[3],
            A_out = cols[4], u_out = cols[5], A_trace = cols[6], u_trace = cols[7],
            resid_norm = cols[8])
end

"""
    volume_balance(sol, w)

For `track_volume = true`: the maximum drift of `V_h + C p_C - V_in + V_R2` over
the saved states (absolute, and relative to the inflow volume). `V_h` is the LGL
quadrature of `A`. The coupled scheme conserves it to round-off.
"""
function volume_balance(sol, w::SemidiscretizationWindkessel)
    _, _, dg, cache = Trixi.mesh_equations_solver_cache(w.semi)
    weights = dg.basis.weights
    x = cache.elements.node_coordinates
    function J(y)
        u = Trixi.wrap_array(dg_state(y, w), w.semi)
        V = zero(eltype(y))
        for element in Trixi.eachelement(dg, cache)
            half = (x[1, end, element] - x[1, 1, element]) / 2
            for i in Trixi.eachnode(dg)
                V += half * weights[i] * u[1, i, element]
            end
        end
        n = w.n
        return V + w.bc.C * y[n + 1] - y[n + 2] + y[n + 3]
    end
    J0 = J(sol.u[1])
    drift = maximum(abs(J(y) - J0) for y in sol.u)
    return (drift = drift, relative = drift / max(abs(sol.u[end][w.n + 2]), eps()))
end
