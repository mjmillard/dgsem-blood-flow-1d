# IndicatorHennemannGassnerT: smooth vs discontinuous data, and bit-for-bit
# agreement with Trixi's IndicatorHennemannGassner at the paper's threshold.
using Test
using Trixi
using Random: MersenneTwister
using BloodFlowDGSEM

const EQ_IND = ArteryFlowEquations1D(1.06, 0.035, 1.1, pi * 0.7688^2, 0.078, 0.4e6, 0.5)
const LENGTH = 500.0
const IND_KWARGS = (; alpha_max = 0.5, alpha_min = 0.001, alpha_smooth = true,
                    variable = area)

function dg_cache(N)
    solver = DGSEM(LobattoLegendreBasis(N), flux_es)
    mesh = TreeMesh(0.0, LENGTH; initial_refinement_level = 4, periodicity = true)
    ic(x, t, eqs) = SVector(eqs.A0, zero(eqs.A0))
    semi = SemidiscretizationHyperbolic(mesh, EQ_IND, ic, solver;
                                        boundary_conditions = boundary_condition_periodic)
    return Trixi.mesh_equations_solver_cache(semi)
end

function state(dg, cache, Afun)
    x = cache.elements.node_coordinates
    u = zeros(nvariables(EQ_IND), nnodes(dg), Trixi.nelements(dg, cache))
    u[1, :, :] .= Afun.(x[1, :, :])
    return u
end

A_smooth(x) = EQ_IND.A0 * (1 + 0.05 * sin(2pi * x / LENGTH))
A_step(x) = x < LENGTH / 2 ? EQ_IND.A0 : 1.6 * EQ_IND.A0

# N = 1 is excluded here: the modal energy ratio is degenerate for two nodes.
@testset "smooth vs discontinuous (N = $N)" for N in (2, 3)
    mesh, eqs, dg, cache = dg_cache(N)
    ind = IndicatorHennemannGassnerT(EQ_IND, dg.basis; T = hg_paper_threshold(dg.basis),
                                     IND_KWARGS...)
    @test maximum(ind(state(dg, cache, A_smooth), mesh, eqs, dg, cache)) <= 0.001 + 1e-8
    @test maximum(ind(state(dg, cache, A_step), mesh, eqs, dg, cache)) > 0.1
end

@testset "bit-for-bit with IndicatorHennemannGassner (N = $N)" for N in 1:4
    rng = MersenneTwister(20260709 + N)
    mesh, eqs, dg, cache = dg_cache(N)
    ind_trixi = IndicatorHennemannGassner(EQ_IND, dg.basis; IND_KWARGS...)
    ind_T = IndicatorHennemannGassnerT(EQ_IND, dg.basis; IND_KWARGS...)   # default T = T(N)
    states = [state(dg, cache, A_smooth), state(dg, cache, A_step)]
    for _ in 1:3
        u = zeros(nvariables(EQ_IND), nnodes(dg), Trixi.nelements(dg, cache))
        u[1, :, :] .= EQ_IND.A0 .* (0.5 .+ rand(rng, size(u)[2:3]...))
        u[2, :, :] .= randn(rng, size(u)[2:3]...)
        push!(states, u)
    end
    for u in states
        @test copy(ind_T(u, mesh, eqs, dg, cache)) == copy(ind_trixi(u, mesh, eqs, dg, cache))
    end
end
