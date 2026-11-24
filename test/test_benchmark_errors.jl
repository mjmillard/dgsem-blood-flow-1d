# The error routine behind the carotid and aorta tables: Boileau et al. Eqs.
# (10)-(13) on a hand case, the phase rule, and the reproduction of every
# published entry of their Tables IV and VI from the supplied waveforms.
using Test
using DelimitedFiles: readdlm
using BloodFlowDGSEM
using BloodFlowDGSEM: boileau_errors, ref_anchor, load_benchmark_cycles,
                      published_dcg_cycle, BOILEAU_3D_OFFSET, BOILEAU_PUBLISHED,
                      BENCHMARK_METHODS, BENCHMARK_QUANTITIES, BENCHMARK_POINTWISE,
                      ERROR_METRICS

# The published values have two decimals; FVM still drifts by up to 0.04 % from
# cycle to cycle, which puts a few of its entries up to 0.011 points off.
const TOL = 0.012

@testset "Eqs. (10)-(13) on a hand case" begin
    ref = [1.0, 2.0]
    num = [1.1, 2.0]
    e = boileau_errors(num, ref; pointwise = true)
    @test e.rms ≈ sqrt(0.1^2 / 2)
    @test e.max ≈ 0.1
    @test e.sys ≈ 0.0 atol = 1e-15
    @test e.dias ≈ 0.1
    e = boileau_errors(num, ref; pointwise = false)
    @test e.max ≈ 0.05
    @test e.dias ≈ 0.05
end

@testset "phase rule" begin
    @test ref_anchor(1.1, 8.8, BOILEAU_3D_OFFSET["carotid"]) ≈ 0.056
    @test ref_anchor(0.955, 19.045, BOILEAU_3D_OFFSET["aorta"]) ≈ 0.002
    @test ref_anchor(0.955, 19.045, 0.055) ≈ 0.0 atol = 1e-9
end

@testset "$case: published $(BOILEAU_PUBLISHED[case].table)" for (case, T, t_ini) in
                                                                (("carotid", 1.1, 8.8),
                                                                 ("aorta", 0.955, 19.045))
    r = readdlm(joinpath(data_dir(case), "3D.txt"))
    tref = r[:, 1]
    ref = (p = r[:, 2], Q = r[:, 3], dr = r[:, 4], dp = r[:, 5])
    pub = BOILEAU_PUBLISHED[case]
    errs(x) = map((num, rv, pw) -> boileau_errors(num, rv; pointwise = pw), x, ref,
                  BENCHMARK_POINTWISE)

    anchor = ref_anchor(T, t_ini, BOILEAU_3D_OFFSET[case])
    codes, bench = load_benchmark_cycles(data_dir(case), T, anchor, tref)
    dcg = published_dcg_cycle(bench, T, anchor, tref)
    for (i, m) in enumerate(BENCHMARK_METHODS)
        E = errs(m == "DCG" ? dcg : codes[m])
        for q in BENCHMARK_QUANTITIES, (k, _) in ERROR_METRICS
            if case == "aorta" && m == "STM" && q === :Q && k === :dias
                @test abs(100 * E[q][k] - 2.95) <= TOL   # printed as 12.95
            else
                @test abs(100 * E[q][k] - getfield(pub, k)[q][i]) <= TOL
            end
        end
    end
    # The paper's DCG column is not DCG at the common phase.
    if case == "aorta"
        @test abs(100 * errs(codes["DCG"]).Q.max - pub.max.Q[1]) > 1.0
    end
end
