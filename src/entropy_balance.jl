# Numerical entropy production on a non-periodic domain without source terms.
# The semidiscrete energy balance is
#
#     dE/dt = Q_neg - Q_pos + Π_num,
#
# with boundary entropy fluxes Q_b = v(u_b)ᵀ f*_b - ψ(u_b), entropy variables v,
# flux potential ψ = ρ u κ A^{3/2} and f*_b the boundary numerical flux. An entropy
# stable scheme has Π_num ≤ 0.

@inline function entropy_flux_potential(u, equations::ArteryFlowEquations1D)
    A, Q = u
    vel = Q / A
    P = equations.kappa * A^(3 / 2)
    return equations.rho * vel * P
end

mutable struct EntropyBoundaryBalance{Semi, SF, BCN, BCP, RealT}
    semi::Semi
    surface_flux::SF
    bc_x_neg::BCN
    bc_x_pos::BCP
    atol::RealT
    rtol::RealT
    t::Vector{RealT}
    dEdt::Vector{RealT}
    Q_neg::Vector{RealT}
    Q_pos::Vector{RealT}
    production::Vector{RealT}
end

# (dE/dt, Q_neg, Q_pos, Π_num) at state u_ode. Calls rhs!, which also overwrites
# the shock indicator's cached blending factor.
function entropy_balance(cb::EntropyBoundaryBalance, u_ode, t)
    semi = cb.semi
    _, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)

    du_ode = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, t)
    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(du_ode, semi)

    weights = dg.basis.weights
    x_nodes = cache.elements.node_coordinates
    dEdt = zero(cb.atol)
    xmin, xmax = typemax(cb.atol), typemin(cb.atol)
    el_L, i_L, el_R, i_R = 1, 1, 1, 1

    for element in Trixi.eachelement(dg, cache), i in Trixi.eachnode(dg)
        u_node = Trixi.get_node_vars(u, equations, dg, i, element)
        du_node = Trixi.get_node_vars(du, equations, dg, i, element)
        J = inv(cache.elements.inverse_jacobian[element])
        dEdt += weights[i] * J * dot(cons2entropy(u_node, equations), du_node)

        x = x_nodes[1, i, element]
        if x < xmin
            xmin, el_L, i_L = x, element, i
        end
        if x > xmax
            xmax, el_R, i_R = x, element, i
        end
    end

    # Rebuild the boundary fluxes Trixi used from the same condition objects.
    u_L = Trixi.get_node_vars(u, equations, dg, i_L, el_L)
    u_R = Trixi.get_node_vars(u, equations, dg, i_R, el_R)
    f_neg = cb.bc_x_neg(u_L, 1, 1, xmin, t, cb.surface_flux, equations)
    f_pos = cb.bc_x_pos(u_R, 1, 2, xmax, t, cb.surface_flux, equations)
    Q_neg = dot(cons2entropy(u_L, equations), f_neg) - entropy_flux_potential(u_L, equations)
    Q_pos = dot(cons2entropy(u_R, equations), f_pos) - entropy_flux_potential(u_R, equations)

    return dEdt, Q_neg, Q_pos, dEdt - Q_neg + Q_pos
end

function (cb::EntropyBoundaryBalance)(integrator)
    dEdt, Q_neg, Q_pos, production = entropy_balance(cb, integrator.u, integrator.t)
    tol = cb.atol + cb.rtol * max(abs(dEdt), abs(Q_neg), abs(Q_pos))
    push!(cb.t, integrator.t)
    push!(cb.dEdt, dEdt)
    push!(cb.Q_neg, Q_neg)
    push!(cb.Q_pos, Q_pos)
    push!(cb.production, production)
    production > tol && @warn "Positive numerical entropy production" t=integrator.t production
    u_modified!(integrator, false)
    return nothing
end

"""
    EntropyBoundaryBalanceCallback(semi; surface_flux, boundary_condition_x_neg,
                                   boundary_condition_x_pos, interval = 100)

Records `Π_num` every `interval` accepted steps and at the start. Pass the same
surface flux and boundary-condition objects used to build `semi`, which must
have no source terms. Retrieve the data with `entropy_balance_history(cb)`.
"""
function EntropyBoundaryBalanceCallback(semi; surface_flux, boundary_condition_x_neg,
                                        boundary_condition_x_pos, interval = 100,
                                        atol = 1.0e-12, rtol = 1.0e-10)
    _, equations, _, _ = Trixi.mesh_equations_solver_cache(semi)
    RealT = typeof(equations.rho)
    balance = EntropyBoundaryBalance(semi, surface_flux, boundary_condition_x_neg,
                                     boundary_condition_x_pos, RealT(atol), RealT(rtol),
                                     RealT[], RealT[], RealT[], RealT[], RealT[])
    condition(u, t, integrator) = integrator.stats.naccept % interval == 0
    return DiscreteCallback(condition, balance;
                            initialize = (cb, u, t, integrator) -> cb.affect!(integrator),
                            save_positions = (false, false))
end

entropy_balance_history(cb::DiscreteCallback) = cb.affect!
