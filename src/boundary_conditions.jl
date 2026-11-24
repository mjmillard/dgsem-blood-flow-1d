# Boundary conditions for ArteryFlowEquations1D on a 1D TreeMesh. Trixi calls the
# x_neg boundary with direction = 1 and x_pos with direction = 2. Each condition
# builds an exterior state u_b from the interior trace and returns the surface
# flux between the two, ordered left to right as in BoundaryConditionDirichlet.

@inline function boundary_flux(u_boundary, u_inner, orientation, direction,
                               surface_flux_function, equations)
    if iseven(direction)    # x_pos: interior on the left
        return surface_flux_function(u_inner, u_boundary, orientation, equations)
    else                    # x_neg: interior on the right
        return surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
end

function newton_solve(nlfunc, L0, p)
    sol = solve(NonlinearProblem(nlfunc, L0, p), SimpleNewtonRaphson();
                abstol = 1.0e-6, reltol = 1.0e-6)
    SciMLBase.successful_retcode(sol) || error("boundary Newton solve failed: $(sol.retcode)")
    return sol.u
end

"""
    BoundaryConditionInflowRiemann(Q_in)

Inlet (x_neg) condition prescribing the flow rate `Q_in(t)`. The outgoing
invariant `u - 4c` is extrapolated from the interior, so the boundary area solves

    Q_in(t) / A_b - 4 c(A_b) = u_i - 4 c(A_i).

Newton runs in `L = log A_b`, which keeps `A_b > 0` at every iterate.
"""
struct BoundaryConditionInflowRiemann{F}
    Q_in::F
end

# g(L) = Q e^{-L} - 4 c(e^L) - W, with c(A) = s A^{1/4}, s = √(1.5 κ); p = (Q, W, s).
@inline function inflow_residual(L, p)
    Q, W2, s = p
    return Q * exp(-L) - 4 * s * exp(0.25 * L) - W2
end

@inline function inflow_jacobian(L, p)
    Q, _, s = p
    return -Q * exp(-L) - s * exp(0.25 * L)
end

const inflow_nlfunc = NonlinearFunction(inflow_residual; jac = inflow_jacobian)

@inline function (bc::BoundaryConditionInflowRiemann)(u_inner, orientation, direction, x, t,
                                                      surface_flux_function,
                                                      equations::ArteryFlowEquations1D)
    Q_b = bc.Q_in(t)
    A_i, u_i = cons2prim(u_inner, equations)
    W2 = u_i - 4 * wave_speed(A_i, equations)
    s = sqrt(1.5 * equations.kappa)
    A_b = exp(newton_solve(inflow_nlfunc, log(A_i), (Q_b, W2, s)))
    return boundary_flux(SVector(A_b, Q_b), u_inner, orientation, direction,
                         surface_flux_function, equations)
end

"""
    BoundaryConditionOutflowNonReflecting(u_ref)

Outlet (x_pos) condition: the outgoing invariant `W1 = u + 4c` is extrapolated
from the interior and the incoming one `W2 = u - 4c` is taken from the reference
state `u_ref`. Then `u_b = (W1 + W2)/2`, `c_b = (W1 - W2)/8` and
`A_b = (c_b² / (1.5 κ))²`.
"""
struct BoundaryConditionOutflowNonReflecting{V}
    u_ref::V
end

@inline function (bc::BoundaryConditionOutflowNonReflecting)(u_inner, orientation, direction,
                                                             x, t, surface_flux_function,
                                                             equations::ArteryFlowEquations1D)
    A_i, u_i = cons2prim(u_inner, equations)
    W1 = u_i + 4 * wave_speed(A_i, equations)
    A_ref, u_ref = cons2prim(bc.u_ref, equations)
    W2 = u_ref - 4 * wave_speed(A_ref, equations)

    u_b = 0.5 * (W1 + W2)
    c_b = 0.125 * (W1 - W2)
    A_b = (c_b^2 / (1.5 * equations.kappa))^2
    return boundary_flux(SVector(A_b, A_b * u_b), u_inner, orientation, direction,
                         surface_flux_function, equations)
end

"""
    BoundaryConditionFixedAreaReflecting()

Characteristic condition valid at either end. With `W₁,₂ = u ± 4(c - c₀)`, the
incoming characteristic is set equal to the outgoing one, which gives `c = c₀`
(so `A = A0`) while the velocity follows the outgoing characteristic.
"""
struct BoundaryConditionFixedAreaReflecting end

@inline function (::BoundaryConditionFixedAreaReflecting)(u_inner, orientation, direction,
                                                          x, t, surface_flux_function,
                                                          equations::ArteryFlowEquations1D)
    A_i, Q_i = u_inner
    u_i = Q_i / A_i
    c_i = wave_speed(A_i, equations)
    c0 = wave_speed(equations.A0, equations)

    if iseven(direction)    # x_pos: W1 outgoing
        W1 = u_i + 4 * (c_i - c0)
        W2 = W1
    else                    # x_neg: W2 outgoing
        W2 = u_i - 4 * (c_i - c0)
        W1 = W2
    end

    # Kept in this form (rather than A_b = A0) so the result matches bit for bit.
    u_b = (W1 + W2) / 2
    c_b = c0 + (W1 - W2) / 8
    A_b = (c_b^2 / (1.5 * equations.kappa))^2
    return boundary_flux(SVector(A_b, A_b * u_b), u_inner, orientation, direction,
                         surface_flux_function, equations)
end

"""
    BoundaryConditionWindkessel3Element(R1, R2, C; p_out = 0.0, p_C0)

Three-element (RCR) Windkessel outlet (x_pos): proximal resistance `R1` in series
with `R2 ∥ C`, draining to `p_out`. The capacitor pressure obeys

    C dp_C/dt = Q_J - (p_C - p_out)/R2.

The exterior state `U_b = (A_b, A_b u_b)` takes `u_b = u_i + 4c(A_i) - 4c(A_b)`
from the outgoing invariant, and `A_b` solves the resistor relation written with
the scheme's own numerical mass flux,

    p(A_b) - p_C - R1 F̂₁(U_i, U_b) = 0.

The capacitor current is that same `Q_J = F̂₁`, so the volume leaving the 1D
domain is exactly the volume entering the 0D model. `p_C` is part of the ODE
state (see [`semidiscretize_windkessel`](@ref)), which writes the current stage's
value into `bc.p_C` before every right-hand-side evaluation; the functor only
reads it. `pressure` includes `p_ext + p_d`, so `p_C0` and `p_out` must use the
same absolute convention.
"""
mutable struct BoundaryConditionWindkessel3Element{RealT}
    R1::RealT
    R2::RealT
    C::RealT
    p_out::RealT
    p_C0::RealT
    p_C::RealT
end

function BoundaryConditionWindkessel3Element(R1, R2, C; p_out = 0.0, p_C0)
    R1, R2, C, p_out, p_C0 = promote(R1, R2, C, p_out, p_C0)
    return BoundaryConditionWindkessel3Element(R1, R2, C, p_out, p_C0, p_C0)
end

"Characteristic impedance `ρ c(A0) / A0`."
characteristic_impedance(equations) = equations.rho * wave_speed(equations.A0, equations) /
                                      equations.A0

@inline function windkessel_exterior_state(L, u_inner, equations::ArteryFlowEquations1D)
    A_i, u_i = cons2prim(u_inner, equations)
    W = u_i + 4 * wave_speed(A_i, equations)
    A_b = exp(L)
    u_b = W - 4 * wave_speed(A_b, equations)
    return SVector(A_b, A_b * u_b)
end

# Resistor relation divided by R1, so the Newton tolerances are in flow units.
@inline function windkessel_residual(L, p)
    u_inner, R1, p_C, orientation, surface_flux, equations = p
    U_b = windkessel_exterior_state(L, u_inner, equations)
    F1 = surface_flux(u_inner, U_b, orientation, equations)[1]
    return F1 - (pressure(U_b, equations) - p_C) / R1
end

# No analytic Jacobian: SimpleNewtonRaphson differentiates through the surface
# flux with ForwardDiff.
const windkessel_nlfunc = NonlinearFunction(windkessel_residual)

@inline function windkessel_exterior(bc::BoundaryConditionWindkessel3Element, u_inner,
                                     orientation, surface_flux, equations)
    p = (u_inner, bc.R1, bc.p_C, orientation, surface_flux, equations)
    L = newton_solve(windkessel_nlfunc, log(u_inner[1]), p)
    return windkessel_exterior_state(L, u_inner, equations)
end

@inline function (bc::BoundaryConditionWindkessel3Element)(u_inner, orientation, direction,
                                                           x, t, surface_flux_function,
                                                           equations::ArteryFlowEquations1D)
    u_b = windkessel_exterior(bc, u_inner, orientation, surface_flux_function, equations)
    return boundary_flux(u_b, u_inner, orientation, direction, surface_flux_function,
                         equations)
end
