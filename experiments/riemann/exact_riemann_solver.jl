# Exact solution of the Riemann problem for
#
#     A_t + (A u)_x = 0,    (A u)_t + (A u² + κ A^{3/2})_x = 0,
#
# following Toro, "Computational Bodily Fluid Dynamics" (2025), Chapter 9
# (Theorem 9.1). Toro's passive scalar is omitted, so A and u are continuous
# across the contact and the solution has two waves (patterns "RS", "SS", ...).
# Toro calls the tube-law constant γ; here it is κ, the same number as the
# `kappa` field of ArteryFlowEquations1D at ν = 1/2. Plain Julia, no Trixi.

"Wave speed `c(A) = √(3/2 κ √A)`, Eq. (9.8)."
@inline c_wave(A, kappa) = sqrt(1.5 * kappa * sqrt(A))

"Inverse of `c_wave`: `A = (2c²/(3κ))²`."
@inline A_from_c(c, kappa) = (2 * c^2 / (3 * kappa))^2

# Wave-curve function f_K(A; A_K) of Eqs. (9.68)-(9.69) and its derivative: the
# velocity change across the K-wave. Rarefaction (A ≤ A_K): f = 4(c(A) - c(A_K)).
# Shock (A > A_K): f = √g, g = κ (A - A_K)(A^{3/2} - A_K^{3/2}) / (A_K A).
@inline function f_wave(A, AK, kappa)
    if A <= AK
        c = c_wave(A, kappa)
        return 4 * (c - c_wave(AK, kappa)), c / A
    else
        sA = sqrt(A)
        sAK = sqrt(AK)
        g = kappa * (A - AK) * (A * sA - AK * sAK) / (AK * A)
        dg = kappa / AK * (1.5 * sA - 0.5 * AK / sA - AK * sAK * sAK * sAK / (A * A))
        return sqrt(g), dg / (2 * sqrt(g))
    end
end

const NEWTON_TOL = 1.0e-6     # relative change, Eq. (9.75)
const NEWTON_MAXIT = 100

# Star state from f(A) = f_L + f_R + (u_R - u_L) = 0 (Eq. 9.67), which is strictly
# increasing in A. Newton (Eq. 9.74) from the two-rarefaction guess (Eqs.
# 9.38/9.40), safeguarded by bisection inside a bracket so that no iterate leaves
# A > 0. Then u* from Eq. (9.70).
function solve_star(AL, uL, AR, uR, kappa)
    cL = c_wave(AL, kappa)
    cR = c_wave(AR, kappa)
    du = uR - uL
    du >= 4 * (cL + cR) && error("solve_star: the data generate vacuum")

    f(A) = first(f_wave(A, AL, kappa)) + first(f_wave(A, AR, kappa)) + du

    c_guess = 0.5 * (cL + cR) - 0.125 * du
    A = A_from_c(c_guess, kappa)

    lo, hi = A, A
    if f(A) > 0
        while f(lo) > 0
            lo *= 0.5
        end
    else
        while f(hi) < 0
            hi *= 2
        end
    end

    converged = false
    for _ in 1:NEWTON_MAXIT
        fA, dfA = f_wave(A, AL, kappa)
        fB, dfB = f_wave(A, AR, kappa)
        fval = fA + fB + du
        dval = dfA + dfB
        fval < 0 ? (lo = max(lo, A)) : (hi = min(hi, A))

        A_new = A - fval / dval
        if !(isfinite(A_new)) || A_new <= 0 || A_new <= lo || A_new >= hi
            A_new = 0.5 * (lo + hi)
        end
        change = abs(A_new - A) / (0.5 * abs(A_new + A))
        A = A_new
        if change < NEWTON_TOL
            converged = true
            break
        end
    end
    converged || error("solve_star: no convergence")

    fL = first(f_wave(A, AL, kappa))
    fR = first(f_wave(A, AR, kappa))
    u_star = 0.5 * (uL + uR) + 0.5 * (fR - fL)
    return A, u_star
end

"""
    RiemannSolution(AL, uL, AR, uR, x0, kappa)

Exact solution for left state `(AL, uL)` on `x < x0` and right state `(AR, uR)`.
A K-wave is a rarefaction if `A* ≤ A_K`, otherwise a shock with speed from the
Rankine-Hugoniot mass flux `M_K = √(κ A_K A* (A*^{3/2} - A_K^{3/2}) / (A* - A_K))`
(Eqs. 9.53, 9.65). For a shock `head == tail` is the shock speed; for a
rarefaction they bound the fan.
"""
struct RiemannSolution
    AL::Float64
    uL::Float64
    AR::Float64
    uR::Float64
    x0::Float64
    kappa::Float64
    A_star::Float64
    u_star::Float64
    c_star::Float64
    left_wave::Symbol
    right_wave::Symbol
    SL_head::Float64
    SL_tail::Float64
    SR_head::Float64
    SR_tail::Float64
end

function RiemannSolution(AL, uL, AR, uR, x0, kappa)
    A_star, u_star = solve_star(AL, uL, AR, uR, kappa)
    c_star = c_wave(A_star, kappa)

    if A_star <= AL
        left_wave = :rarefaction
        SL_head = uL - c_wave(AL, kappa)
        SL_tail = u_star - c_star
    else
        left_wave = :shock
        ML = sqrt(kappa * AL * A_star * (A_star^1.5 - AL^1.5) / (A_star - AL))
        SL_head = SL_tail = uL - ML / AL
    end

    if A_star <= AR
        right_wave = :rarefaction
        SR_head = uR + c_wave(AR, kappa)
        SR_tail = u_star + c_star
    else
        right_wave = :shock
        MR = sqrt(kappa * AR * A_star * (A_star^1.5 - AR^1.5) / (A_star - AR))
        SR_head = SR_tail = uR + MR / AR
    end

    return RiemannSolution(AL, uL, AR, uR, x0, kappa, A_star, u_star, c_star,
                           left_wave, right_wave, SL_head, SL_tail, SR_head, SR_tail)
end

# Pattern in Toro's notation; a fan straddling x/t = 0 is "R(sonic)".
function wave_pattern(sol::RiemannSolution)
    l = sol.left_wave === :shock ? "S" : sol.SL_head < 0 < sol.SL_tail ? "R(sonic)" : "R"
    r = sol.right_wave === :shock ? "S" : sol.SR_tail < 0 < sol.SR_head ? "R(sonic)" : "R"
    return l * r
end

# Fastest characteristic speed anywhere in the solution.
function max_signal_speed(sol::RiemannSolution)
    return max(abs(sol.uL) + c_wave(sol.AL, sol.kappa),
               abs(sol.uR) + c_wave(sol.AR, sol.kappa),
               abs(sol.u_star) + sol.c_star,
               abs(sol.SL_head), abs(sol.SL_tail),
               abs(sol.SR_head), abs(sol.SR_tail))
end

# Speed of the outermost disturbance.
wave_reach_speed(sol::RiemannSolution) = max(abs(sol.SL_head), abs(sol.SR_head))

# Self-similar solution at xi = (x - x0)/t (Eqs. 9.81-9.84). Inside a fan the
# Riemann invariant gives u = (u_L + 4c_L + 4ξ)/5, c = (u_L + 4c_L - ξ)/5 (left)
# and u = (u_R - 4c_R + 4ξ)/5, c = (-u_R + 4c_R + ξ)/5 (right).
function sample(sol::RiemannSolution, xi)
    if xi <= sol.u_star
        if sol.left_wave === :shock
            return xi <= sol.SL_head ? (sol.AL, sol.uL) : (sol.A_star, sol.u_star)
        elseif xi <= sol.SL_head
            return (sol.AL, sol.uL)
        elseif xi >= sol.SL_tail
            return (sol.A_star, sol.u_star)
        else
            w = sol.uL + 4 * c_wave(sol.AL, sol.kappa)
            u = (w + 4 * xi) / 5
            c = (w - xi) / 5
            return (A_from_c(c, sol.kappa), u)
        end
    else
        if sol.right_wave === :shock
            return xi >= sol.SR_head ? (sol.AR, sol.uR) : (sol.A_star, sol.u_star)
        elseif xi >= sol.SR_head
            return (sol.AR, sol.uR)
        elseif xi <= sol.SR_tail
            return (sol.A_star, sol.u_star)
        else
            w = -sol.uR + 4 * c_wave(sol.AR, sol.kappa)
            u = (-w + 4 * xi) / 5
            c = (w + xi) / 5
            return (A_from_c(c, sol.kappa), u)
        end
    end
end

# Solution at (x, t); t = 0 is a separate branch because x0 lies on a mesh node.
@inline function solution_at(sol::RiemannSolution, x, t)
    t <= 0 && return x < sol.x0 ? (sol.AL, sol.uL) : (sol.AR, sol.uR)
    return sample(sol, (x - sol.x0) / t)
end

function exact_riemann_profile(sol::RiemannSolution, xs, t)
    A = similar(collect(Float64, xs))
    u = similar(A)
    for (i, x) in enumerate(xs)
        A[i], u[i] = solution_at(sol, x, t)
    end
    return A, u
end

"κ = (4√π/9) E h0 / (ρ A0), Eq. (9.5) at ν = 1/2; defaults are Table 9.1 in CGS."
toro_kappa(; E = 3.0e6, h0 = 0.05, rho = 1.0, A0 = 3.14) =
    (4 * sqrt(pi) / 9) * E * h0 / (rho * A0)

# Table 9.2 in CGS (areas in cm², velocities in cm/s), with the book's patterns.
const TORO_TESTS = (
    (id = 1, AL = 3.5, uL = 0.0, AR = 3.0, uR = 0.0, Tout = 0.05, pattern = "RS"),
    (id = 2, AL = 10.0, uL = 0.0, AR = 1.0, uR = 0.0, Tout = 0.04, pattern = "R(sonic)S"),
    (id = 3, AL = 3.14, uL = -50.0, AR = 3.14, uR = 50.0, Tout = 0.05, pattern = "RR"),
    (id = 4, AL = 3.14, uL = 50.0, AR = 3.14, uR = -50.0, Tout = 0.05, pattern = "SS"),
)
