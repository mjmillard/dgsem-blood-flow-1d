"""
Entropy-stable DGSEM for one-dimensional blood flow in compliant arteries, built
on Trixi.jl.
"""
module BloodFlowDGSEM

using DelimitedFiles: readdlm
using LaTeXStrings: @L_str
using LinearAlgebra: dot
using OrdinaryDiffEqLowStorageRK: RDPK3Sp510
using Plots: Plots, plot, plot!, hline!, savefig
using Printf: @printf, @sprintf
using SciMLBase: SciMLBase, DiscreteCallback, ODEProblem, u_modified!
using SimpleNonlinearSolve: NonlinearFunction, NonlinearProblem, SimpleNewtonRaphson, solve
using StaticArrays: SVector
using Statistics: mean
using Trixi
using Trixi: AbstractEquations, AbstractIndicator, LobattoLegendreBasis, DGSEM,
             nelements, eachelement, nnodes, real, flux_ec, pressure

include("equations.jl")
include("boundary_conditions.jl")
include("windkessel.jl")
include("output.jl")
include("shock_indicator.jl")
include("entropy_balance.jl")
include("benchmark.jl")

"`results/<name>/` in the repository root, created if needed."
results_dir(name) = mkpath(joinpath(pkgdir(@__MODULE__), "results", name))

"`data/boileau2015/<case>/` in the repository root."
data_dir(case) = joinpath(pkgdir(@__MODULE__), "data", "boileau2015", case)

export ArteryFlowEquations1D, area, pressure, wave_speed, flux_ec, flux_es,
       source_terms_friction
export BoundaryConditionInflowRiemann, BoundaryConditionOutflowNonReflecting,
       BoundaryConditionFixedAreaReflecting, BoundaryConditionWindkessel3Element,
       characteristic_impedance
export semidiscretize_windkessel, windkessel_rhs!, windkessel_history, volume_balance,
       dg_state
export IndicatorHennemannGassnerT, hg_paper_threshold, ShockIndicatorMonitorCallback,
       shock_indicator_history, plot_shock_indicator
export EntropyBoundaryBalanceCallback, entropy_balance_history
export thesis_plot_style!, thesis_savefig, FIG_DPI, GUIDEFONT, TICKFONT, LEGENDFONT
export TableColumn, TableSpec, emit_tables, print_ascii
export load_benchmark_methods, evaluate_dg, cfl_dt_recorder, resolution_table,
       run_benchmark
export results_dir, data_dir

end
