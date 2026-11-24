# Upper thoracic aorta benchmark (Boileau et al. 2015, Section 4.3, Table V),
# thesis Section 5.3. Writes results/aorta/aorta_{midpoint_pressure,
# inlet_outlet_pressure_difference, midpoint_flow_rate, radius_change_from_diastole,
# windkessel}.png and aorta_errors.md.
using BloodFlowDGSEM

# CGS units. Inflow: 500 times the 40-harmonic Fourier series of the supplement's
# inflow.txt, evaluated at t + 0.002 s so that the 3D-aligned cycle is the last
# simulated one.
const AORTA = (name = "aorta",
    title = "Upper thoracic aorta -- midpoint errors vs the 3D reference (Boileau et al., Table VI)",
    datadir = data_dir("aorta"),
    L = 24.137, A_d = 4.5239, h0 = 0.12, rho = 1.06, E = 4.0e6, mu = 0.04,
    p_d = 2.84e5 / 3, zeta = 9,
    R1 = 1.1752e7 * 1e-5, R2 = 1.1167e8 * 1e-5, C = 1.0163e-8 * 1e5,
    T = 0.955, t_ini = 19.045, inflow_shift = 0.002, Q_mean = 103.085,
    inflow_dc = 0.20617, inflow_scale = 500.0,
    inflow_amp = (0.37759, 0.2804, 0.15337, -0.049889, 0.038107, -0.041699,
                  -0.020754, 0.013367, -0.021983, -0.013072, 0.0037028, -0.013973,
                  -0.012423, 0.0040098, -0.0059704, -0.0073439, 0.0037006, -0.0032069,
                  -0.0048171, 0.0040403, -0.0032409, -0.0032517, 0.0029112, -0.0022708,
                  -0.0021566, 0.0025511, -0.0024448, -0.0019032, 0.0019476, -0.0019477,
                  -0.0014545, 0.0013979, -0.0014305, -0.0010775, 0.0010368, -0.0012162,
                  -0.00095707, 0.00077733, -0.00092407, -0.00079585),
    inflow_phase = (0.59605, -0.35859, -1.2509, 1.3921, -1.1068, 1.3985, 0.72921,
                    -1.5394, 0.95617, -0.022417, -1.4146, 0.77416, -0.46511, 0.95145,
                    0.86369, -0.64769, 0.74663, 0.85926, -1.0306, 0.28009, 1.202,
                    -0.93316, 0.21405, 1.1869, -1.1574, -0.12915, 1.1185, -0.99244,
                    -0.059885, 1.1655, -0.85829, 0.042912, 1.2439, -0.79464, -0.0043058,
                    1.211, -0.66203, 0.25642, 1.3954, -0.49973),
    refine = 1,      # two elements
    ncycles = 15,
    # Headroom above the flow peaks for the legend.
    windkessel_flow_axis = Q -> begin
        Q_lo, Q_hi = extrema(Q)
        (ylims = (Q_lo - 0.05 * (Q_hi - Q_lo), Q_hi + 0.3 * (Q_hi - Q_lo)),
         legend = :topright)
    end)

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark(AORTA; outdir = results_dir("aorta"))
end
