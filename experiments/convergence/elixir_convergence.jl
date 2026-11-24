# Manufactured-solution run for `Trixi.convergence_test` (see convergence.jl),
# which overrides polydeg, initial_refinement_level, cfl and time_integrator.
using Trixi
using OrdinaryDiffEqLowStorageRK
using BloodFlowDGSEM

rho = 1.06
A0 = 1.0
mu = 0.035
alpha = 1.1
h0 = 0.078
E = 0.4 * 10^6
nu = 0.5
equations = ArteryFlowEquations1D(rho, mu, alpha, A0, h0, E, nu)

initial_condition = initial_condition_convergence_test

boundary_conditions = (x_neg = BoundaryConditionFixedAreaReflecting(),
                       x_pos = BoundaryConditionFixedAreaReflecting())

solver = DGSEM(polydeg = 3, surface_flux = flux_es,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ec))

coordinates_min = 0.0
coordinates_max = 20.0
mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level = 6,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms = source_terms_convergence_test,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100,
                                     analysis_errors = [:l2_error, :linf_error])
stepsize_callback = StepsizeCallback(cfl = 0.5)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

# A variable, so that convergence_test can substitute the integrator.
# StepsizeCallback requires adaptive = false.
time_integrator = CarpenterKennedy2N54(williamson_condition = false)

sol = solve(ode, time_integrator;
            dt = 1, adaptive = false, ode_default_options()..., callback = callbacks)

summary_callback()
