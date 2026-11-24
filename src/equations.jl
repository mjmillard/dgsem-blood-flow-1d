"""
    ArteryFlowEquations1D(rho, mu, alpha, A0, h0, E, nu; p_ext = 0.0, p_d = 0.0)

One-dimensional blood flow in a compliant artery in conservative variables
`u = (A, Q)` (cross-sectional area, flow rate):

    ∂A/∂t + ∂Q/∂x = 0
    ∂Q/∂t + ∂(Q²/A + κ A^{3/2})/∂x = -K_R Q/A

The momentum-flux correction coefficient `alpha` only enters the friction
coefficient `K_R`; the convective flux uses a flat profile. The tube law is
`p = K_p (√(A/A0) - 1) + p_ext + p_d` with `K_p = √π E h0 / ((1 - ν²) √A0)`, and
`κ = K_p / (3 ρ √A0)`. The offsets `p_ext` and `p_d` enter only [`pressure`](@ref):
the flux, wave speed, friction and entropy variables do not depend on them.

The equations are unit-agnostic; each experiment uses one consistent system.
"""
struct ArteryFlowEquations1D{RealT <: Real} <: AbstractEquations{1, 2}
    rho::RealT
    mu::RealT
    alpha::RealT
    A0::RealT
    h0::RealT
    E::RealT
    nu::RealT
    K_p::RealT
    K_R::RealT
    inv_sqrt_A0::RealT
    kappa::RealT
    p_ext::RealT
    p_d::RealT

    function ArteryFlowEquations1D(rho, mu, alpha, A0, h0, E, nu; p_ext = 0.0, p_d = 0.0)
        inv_sqrt_A0 = inv(sqrt(A0))
        K_p = sqrt(pi) * E * h0 * inv_sqrt_A0 / (1.0 - nu^2)
        # Inviscid flow has no friction; without the short-circuit mu = 0 with
        # alpha = 1 (the Riemann problems) would give 0/0.
        K_R = iszero(mu) ? zero(mu * alpha / rho) :
              2.0 * pi * mu * alpha / (rho * (alpha - 1.0))
        kappa = K_p * inv_sqrt_A0 / (3.0 * rho)

        rho_, mu_, alpha_, A0_, h0_, E_, nu_, K_p_, K_R_, inv_sqrt_A0_, kappa_, p_ext_, p_d_ =
            promote(rho, mu, alpha, A0, h0, E, nu, K_p, K_R, inv_sqrt_A0, kappa, p_ext, p_d)

        return new{typeof(rho_)}(rho_, mu_, alpha_, A0_, h0_, E_, nu_,
                                 K_p_, K_R_, inv_sqrt_A0_, kappa_, p_ext_, p_d_)
    end
end

Trixi.varnames(::typeof(cons2cons), ::ArteryFlowEquations1D) = ("A", "Q")
Trixi.varnames(::typeof(cons2prim), ::ArteryFlowEquations1D) = ("A", "u")

@inline function Trixi.cons2prim(u, equations::ArteryFlowEquations1D)
    A, Q = u
    return SVector(A, Q / A)
end

@inline area(u, equations::ArteryFlowEquations1D) = u[1]

@inline function Trixi.pressure(u, equations::ArteryFlowEquations1D)
    return equations.K_p * (sqrt(u[1] / equations.A0) - 1) + equations.p_ext + equations.p_d
end

"Pulse wave speed `c(A) = √(3/2 κ √A)`."
@inline wave_speed(A, equations::ArteryFlowEquations1D) = sqrt(1.5 * equations.kappa * sqrt(A))

@inline function Trixi.flux(u, orientation::Integer, equations::ArteryFlowEquations1D)
    A, Q = u
    return SVector(Q, Q^2 / A + equations.kappa * A^(3 / 2))
end

@inline function Trixi.max_abs_speeds(u, equations::ArteryFlowEquations1D)
    A, vel = cons2prim(u, equations)
    return (abs(vel) + wave_speed(A, equations),)
end

# λ_max = max(|u| + c) over the two traces; Trixi's `max_abs_speed` falls back to
# this, so it is the estimate `DissipationLocalLaxFriedrichs` uses.
@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                           equations::ArteryFlowEquations1D)
    A_ll, vel_ll = cons2prim(u_ll, equations)
    A_rr, vel_rr = cons2prim(u_rr, equations)
    return max(abs(vel_ll) + wave_speed(A_ll, equations),
               abs(vel_rr) + wave_speed(A_rr, equations))
end

@inline function source_terms_friction(u, x, t, equations::ArteryFlowEquations1D)
    A, Q = u
    return SVector(zero(eltype(u)), -equations.K_R * Q / A)
end

# Manufactured solution A = (1 + t e^{-10t} sin(πx/L))², Q = 0 on [0, L], L = 20.
function Trixi.initial_condition_convergence_test(x, t, equations::ArteryFlowEquations1D)
    L = 20
    A = (1 + t * exp(-10 * t) * sin(pi * x[1] / L))^2
    return SVector(A, zero(A))
end

@inline function Trixi.source_terms_convergence_test(u, x, t,
                                                     equations::ArteryFlowEquations1D)
    L = 20
    A, Q = u
    c = wave_speed(A, equations)
    s1 = -2 * sin(pi * x[1] / L) * exp(-10 * t) * (10 * t - 1) *
         (t * sin(pi * x[1] / L) * exp(-10 * t) + 1)
    s2 = 2 * pi * c^2 * t * cos(pi * x[1] / L) * exp(-10 * t) *
         (t * sin(pi * x[1] / L) * exp(-10 * t) + 1) / L
    return SVector(s1, s2)
end

# Entropy = total energy η(A, Q) = ½ ρ Q²/A + Ψ(A) with Ψ'(A) = p(A) (gauge):
#     Ψ(A) = ⅔ K_p A^{3/2} / √A0 − K_p A = 2 ρ κ A^{3/2} − K_p A.
# η is strictly convex for A > 0 and `cons2entropy` is its gradient. The offsets
# p_ext + p_d would add a linear term to η, which cancels in every jump.
@inline function Trixi.entropy(u, equations::ArteryFlowEquations1D)
    A, Q = u
    return 0.5 * equations.rho * Q^2 / A +
           2 * equations.rho * equations.kappa * A^(3 / 2) - equations.K_p * A
end

@inline function Trixi.cons2entropy(u, equations::ArteryFlowEquations1D)
    A, Q = u
    vel = Q / A
    p = equations.K_p * (sqrt(A) * equations.inv_sqrt_A0 - 1)
    return SVector(p - 0.5 * equations.rho * vel^2, equations.rho * vel)
end

"""
    flux_ec(u_ll, u_rr, orientation, equations::ArteryFlowEquations1D)

Entropy-conservative two-point flux: it satisfies Tadmor's condition
`⟦v⟧ᵀ f_ec = ⟦ψ⟧` exactly, with entropy variables `v` and flux potential
`ψ = ρ u κ A^{3/2}`. Uses the mean `Ã = (A_ll + √(A_ll A_rr) + A_rr)/3`.
"""
@inline function Trixi.flux_ec(u_ll, u_rr, orientation::Integer,
                               equations::ArteryFlowEquations1D)
    A_ll, vel_ll = cons2prim(u_ll, equations)
    A_rr, vel_rr = cons2prim(u_rr, equations)

    A_tilde = 1 / 3 * (A_ll + sqrt(A_ll * A_rr) + A_rr)
    vel_avg = 0.5 * (vel_ll + vel_rr)
    P_avg = 0.5 * (equations.kappa * A_ll^(3 / 2) + equations.kappa * A_rr^(3 / 2))

    return SVector(A_tilde * vel_avg, A_tilde * vel_avg^2 + P_avg)
end

# Entropy-stable surface flux f_es = f_ec − ½ λ_max ⟦u⟧. Its interface entropy
# production is −½ λ_max ⟦∇η⟧ᵀ⟦u⟧ ≤ 0 because ∇η of a strictly convex η is
# monotone, so it is entropy stable on both flow branches without any
# eigenvalue or sonic-point case distinction.
const flux_es = FluxPlusDissipation(flux_ec, DissipationLocalLaxFriedrichs())
