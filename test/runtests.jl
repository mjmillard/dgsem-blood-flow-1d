# julia --project=. test/runtests.jl   (a few minutes, mostly test_windkessel.jl)
using Test

@testset "BloodFlowDGSEM" begin
    for file in ("test_fluxes.jl", "test_shock_indicator.jl", "test_exact_riemann_solver.jl",
                 "test_benchmark_errors.jl", "test_windkessel.jl")
        @testset "$file" begin
            include(file)
        end
    end
end
