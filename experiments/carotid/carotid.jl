# Common carotid artery benchmark (Boileau et al. 2015, Section 4.2, Table III),
# thesis Section 5.2. Writes results/carotid/carotid_{midpoint_pressure,
# inlet_outlet_pressure_difference, midpoint_flow_rate, radius_change_from_diastole,
# windkessel}.png and carotid_errors.md.
using BloodFlowDGSEM

# CGS units. Inflow: the 26-harmonic Fourier series of the supplement's
# inflow.txt (mean 6.5 ml/s), evaluated at t + 0.056 s so that the 3D-aligned
# cycle is the last simulated one.
const CAROTID = (name = "carotid",
    title = "Carotid artery -- midpoint errors vs the 3D reference (Boileau et al., Table IV)",
    datadir = data_dir("carotid"),
    L = 12.6, A_d = 0.28274, h0 = 0.03, rho = 1.06, E = 7.0e6, mu = 0.04,
    p_d = 1.0933e5, zeta = 2,
    R1 = 2.4875e8 * 1e-5, R2 = 1.8697e9 * 1e-5, C = 1.7529e-10 * 1e5,
    T = 1.1, t_ini = 8.8, inflow_shift = 0.056, Q_mean = 6.5,
    inflow_dc = 6.5, inflow_scale = 1.0,
    inflow_amp = (3.294, 1.9262, -1.4219, -0.66627, -0.33933, -0.37914, 0.22396,
                  0.1507, 0.18735, 0.038625, 0.012643, -0.0042453, -0.012781,
                  0.014805, 0.012249, 0.0076502, 0.0030692, -0.0012271, -0.0042581,
                  -0.0069785, 0.0085652, 0.0081881, 0.0056549, 0.0026358, -0.0050868,
                  -0.0085829),
    inflow_phase = (-0.023974, -1.1801, 0.92701, -0.24118, -0.27471, -1.0557, 1.22,
                    1.0984, 0.067483, 0.22262, -0.10093, -1.1044, -1.3739, 1.2797,
                    0.80827, 0.40757, 0.195, -1.1371, -0.92102, -1.2364, 1.4539,
                    0.89599, 0.17623, -1.3003, -0.011056, -0.86463),
    refine = 0,      # one element
    ncycles = 9,
    windkessel_flow_axis = Q -> (;))

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark(CAROTID; outdir = results_dir("carotid"))
end
