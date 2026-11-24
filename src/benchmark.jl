# Comparison with the benchmark of Boileau et al. (2015, doi:10.1002/cnm.2732):
# the 3D reference of Xiao et al. (2014) and the six 1D codes DCG, LCG, FEM, FVM,
# McC and STM, from the paper's data supplement (doi:10.6084/m9.figshare.1586944).
#
# Phase rule. All 1D series, ours and the six codes', are put on the time base of
# the raw inflow.txt Fourier series (the codes' file time; our simulation time
# plus the inflow shift) and sampled at the 3D reference's 1 ms points. The 3D
# t = 0 corresponds to raw-inflow time t_ini + offset (mod T), with t_ini the
# supplement README's "last cardiac cycle" start (carotid 8.8 s, aorta 19.045 s)
# and offset 0.056 s (carotid) / 0.057 s (aorta). At this phase the routine below
# reproduces every published entry of the paper's Tables IV and VI to rounding
# (DCG sampled one row later, see DCG_ROW_LAG). Cross-correlating the six codes
# with the 3D data places the alignment at +54..58 ms, so the 3D phase is not
# identifiable to the millisecond; the paper's convention is used unchanged.
#
# Sampling. The carotid and aorta runs take full CFL steps, save every step, and
# are resampled to the 1 ms grid by cubic Hermite interpolation. In space the DG
# polynomial is evaluated exactly. The supplement files carry about five
# significant digits, which limits the comparison to about 1 Pa in pressure.

const BENCHMARK_METHODS = ("DCG", "LCG", "FEM", "FVM", "McC", "STM")

function load_benchmark_methods(dir::AbstractString; suffix::AbstractString = "")
    return Dict(m => readdlm(joinpath(dir, m * suffix * ".txt"), Float64)
                for m in BENCHMARK_METHODS)
end

# Linear interpolation of (xs, ys) at xq, xs ascending; errors outside the data.
function interp_linear(xs::AbstractVector, ys::AbstractVector, xq::AbstractVector;
                       tol = 1e-9)
    out = similar(xq, Float64)
    for (k, x) in enumerate(xq)
        xs[1] - tol <= x <= xs[end] + tol || throw(DomainError(x, "outside the data"))
        j = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
        w = (x - xs[j]) / (xs[j + 1] - xs[j])
        out[k] = (1 - w) * ys[j] + w * ys[j + 1]
    end
    return out
end

# DG polynomial of variable `var` at the points `xs` (the left element at an
# interface). `u_dg` is the DG part of an ODE state.
function evaluate_dg(u_dg, semi, xs::AbstractVector; var::Integer = 1)
    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    nodes = dg.basis.nodes
    wbary = Trixi.barycentric_weights(nodes)
    x = cache.elements.node_coordinates
    order = sortperm([x[1, 1, e] for e in Trixi.eachelement(dg, cache)])
    lefts = [x[1, 1, e] for e in order]
    u = Trixi.wrap_array(u_dg, semi)
    out = similar(xs, Float64)
    for (k, xq) in enumerate(xs)
        j = clamp(searchsortedfirst(lefts, xq) - 1, 1, length(lefts))
        e = order[j]
        xi = 2 * (xq - x[1, 1, e]) / (x[1, end, e] - x[1, 1, e]) - 1
        ell = Trixi.lagrange_interpolating_polynomials(xi, nodes, wbary)
        out[k] = sum(ell[i] * u[var, i, e] for i in eachindex(ell))
    end
    return out
end

# Records the step size the CFL condition proposes at every accepted step (place
# after the StepsizeCallback). With `tstops` the accepted steps are clipped to the
# output grid; this keeps the unclipped value.
function cfl_dt_recorder()
    dts = Float64[]
    affect!(integrator) = (push!(dts, integrator.dtcache); u_modified!(integrator, false))
    return DiscreteCallback((u, t, integrator) -> true, affect!;
                            save_positions = (false, false)), dts
end

# States at the times `tq` by cubic Hermite interpolation between the stored steps
# of `sol`, with slopes from `f!(du, u, p, t)`. A query on a stored time returns
# that step's value exactly.
function hermite_states(sol, f!, p, tq::AbstractVector)
    ts = sol.t
    du = [similar(u) for u in sol.u]
    for (j, u) in enumerate(sol.u)
        f!(du[j], u, p, ts[j])
    end
    out = Vector{eltype(sol.u)}(undef, length(tq))
    for (k, t) in enumerate(tq)
        j = clamp(searchsortedlast(ts, t), 1, length(ts) - 1)
        h = ts[j + 1] - ts[j]
        s = (t - ts[j]) / h
        h00 = (1 + 2s) * (1 - s)^2
        h10 = s * (1 - s)^2
        h01 = s^2 * (3 - 2s)
        h11 = s^2 * (s - 1)
        out[k] = s == 0 ? copy(sol.u[j]) : s == 1 ? copy(sol.u[j + 1]) :
                 h00 .* sol.u[j] .+ (h10 * h) .* du[j] .+
                 h01 .* sol.u[j + 1] .+ (h11 * h) .* du[j + 1]
    end
    return (t = collect(tq), u = out)
end

const BOILEAU_3D_OFFSET = Dict("carotid" => 0.056, "aorta" => 0.057)

# The paper's DCG column is the DCG series sampled 1 ms later than the others:
# DCG's file alone starts at t = 0.001, and the paper extracted the cycle by row.
const DCG_ROW_LAG = 0.001

# Raw-inflow phase in [0, T) of the 3D series' t = 0.
function ref_anchor(T, t_ini, offset)
    a = mod(t_ini + offset, T)
    return isapprox(a, T; atol = 1e-9) ? zero(a) : a
end

# Start k T + anchor of the last full window of length tau_end inside `t`,
# stepped back `cycles_back` periods.
function last_cycle_start(t::AbstractVector, T, anchor, tau_end; cycles_back::Integer = 0)
    k = floor((t[end] + 1e-9 - anchor - tau_end) / T) - cycles_back
    return k * T + anchor
end

# Supplement columns 2:5: p [kPa], Q [ml/s], dr [mm], dp [kPa].
const BENCHMARK_QUANTITIES = (:p, :Q, :dr, :dp)
# Eqs. (10)-(13): pressure is normalized pointwise, the others by the cycle maximum.
const BENCHMARK_POINTWISE = (p = true, Q = false, dr = false, dp = false)

function cycle_at_ref_times(t_raw::AbstractVector, series::NamedTuple, T, anchor,
                            tref::AbstractVector; cycles_back::Integer = 0)
    t0 = last_cycle_start(t_raw, T, anchor, tref[end]; cycles_back)
    return map(x -> interp_linear(t_raw, x, t0 .+ tref), series[BENCHMARK_QUANTITIES])
end

benchmark_series(d::AbstractMatrix) = (p = d[:, 2], Q = d[:, 3], dr = d[:, 4], dp = d[:, 5])

function load_benchmark_cycles(dir::AbstractString, T, anchor, tref::AbstractVector)
    bench = load_benchmark_methods(dir)
    cycles = Dict(m => cycle_at_ref_times(view(d, :, 1), benchmark_series(d), T, anchor, tref)
                  for (m, d) in bench)
    return cycles, bench
end

function published_dcg_cycle(bench::AbstractDict, T, anchor, tref::AbstractVector)
    d = bench["DCG"]
    return cycle_at_ref_times(view(d, :, 1), benchmark_series(d), T, anchor + DCG_ROW_LAG,
                              tref)
end

# Eqs. (10)-(13) of Boileau et al.: RMS, MAX, systolic and diastolic relative
# errors of `num` against `ref` at common time points. Pressure is normalized
# pointwise (DIAS by min(ref)); Q, dr and dp by max(ref).
function boileau_errors(num, ref; pointwise::Bool)
    rel = (num .- ref) ./ (pointwise ? ref : maximum(ref))
    return (rms = sqrt(sum(abs2, rel) / length(rel)),
            max = maximum(abs, rel),
            sys = (maximum(num) - maximum(ref)) / maximum(ref),
            dias = (minimum(num) - minimum(ref)) / (pointwise ? minimum(ref) : maximum(ref)))
end

# Published values in per cent, columns DCG..STM, signs as printed (Boileau et
# al. Tables IV and VI).
const BOILEAU_PUBLISHED = Dict(
    "carotid" => (table = "Table IV",
        rms  = (p = [0.20, 0.24, 0.23, 0.28, 0.24, 0.24], Q  = [0.39, 0.26, 0.28, 0.29, 0.30, 0.29],
                dr = [0.96, 1.00, 1.00, 1.07, 1.01, 0.98], dp = [4.42, 4.35, 4.27, 4.21, 4.47, 4.47]),
        max  = (p = [0.34, 0.52, 0.50, 0.66, 0.52, 0.53], Q  = [1.23, 1.02, 1.12, 1.07, 1.20, 1.13],
                dr = [1.75, 1.93, 1.88, 2.28, 1.93, 1.96], dp = [16.69, 16.17, 15.88, 15.60, 16.58, 16.90]),
        sys  = (p = [-0.27, -0.27, -0.26, -0.29, -0.27, -0.26], Q = [-0.55, -0.46, -0.54, -0.58, -0.62, -0.55],
                dr = [-1.65, -1.63, -1.61, -1.68, -1.63, -1.63],
                dp = [-15.05, -15.37, -15.16, -14.83, -15.73, -15.76]),
        dias = (p = [0.28, 0.28, 0.27, 0.29, 0.28, 0.27], Q  = [0.25, 0.22, 0.26, 0.24, 0.23, 0.27],
                dr = [0.08, 0.11, 0.08, 0.12, 0.09, 0.08], dp = [4.99, 4.76, 5.03, 4.91, 4.45, 4.85])),
    "aorta" => (table = "Table VI",
        rms  = (p = [1.09, 1.21, 1.09, 1.14, 1.08, 1.12], Q  = [2.55, 2.19, 2.23, 2.17, 2.22, 2.28],
                dr = [2.32, 2.41, 2.33, 2.44, 2.33, 2.90], dp = [7.70, 7.23, 7.23, 7.14, 7.23, 7.52]),
        max  = (p = [3.54, 3.55, 3.26, 3.18, 3.27, 3.30], Q  = [9.13, 7.20, 7.04, 7.07, 7.04, 7.40],
                dr = [7.80, 7.67, 7.32, 7.20, 7.09, 8.04], dp = [31.83, 29.31, 29.15, 29.13, 29.31, 31.19]),
        sys  = (p = [-0.57, -0.32, -0.56, -0.71, -0.55, -0.59], Q = [-5.56, -5.36, -5.63, -5.29, -5.58, -5.56],
                dr = [-2.54, -1.93, -2.50, -2.86, -2.49, -4.48],
                dp = [-8.92, -8.89, -9.29, -8.53, -9.04, -9.70]),
        dias = (p = [0.85, 1.12, 0.88, 0.99, 0.87, 0.85], Q  = [2.75, 2.67, 3.09, 3.48, 2.67, 12.95],
                dr = [1.85, 2.24, 1.89, 2.05, 1.89, 0.00], dp = [7.32, 7.24, 7.60, 6.37, 7.32, 7.47])))

# Boileau et al. Table I: dt [ms], element length Omega_e [cm], CFL = λ dt / Omega_e
# (`nothing` = NA), space / time order.
const BOILEAU_TABLE_I = Dict(
    "single_pulse" => [(scheme = "DCG", dt = 0.1,    h = 2.00,  cfl = 0.18,    space = "5th", time = "2nd"),
                       (scheme = "LCG", dt = 0.01,   h = 0.20,  cfl = 0.03,    space = "2nd", time = "2nd"),
                       (scheme = "FEM", dt = 0.0001, h = 0.25,  cfl = nothing, space = "2nd", time = "1st"),
                       (scheme = "FVM", dt = 1.457,  h = 1.00,  cfl = 0.9,     space = "4th", time = "4th"),
                       (scheme = "McC", dt = 0.25,   h = 0.156, cfl = 0.99,    space = "2nd", time = "2nd"),
                       (scheme = "STM", dt = 0.1,    h = 0.10,  cfl = nothing, space = "1st", time = "2nd")],
    "carotid" => [(scheme = "DCG", dt = 1.0,  h = 12.60, cfl = 0.48,    space = "5th", time = "2nd"),
                  (scheme = "LCG", dt = 0.2,  h = 0.17,  cfl = 0.8,     space = "2nd", time = "2nd"),
                  (scheme = "FEM", dt = 0.1,  h = 1.26,  cfl = nothing, space = "2nd", time = "1st"),
                  (scheme = "FVM", dt = 3.27, h = 2.00,  cfl = 0.9,     space = "2nd", time = "2nd"),
                  (scheme = "McC", dt = 0.5,  h = 0.39,  cfl = 0.87,    space = "2nd", time = "2nd"),
                  (scheme = "STM", dt = 1.0,  h = 1.00,  cfl = nothing, space = "1st", time = "2nd")],
    "aorta"   => [(scheme = "DCG", dt = 0.5,  h = 12.07, cfl = 0.19,    space = "5th", time = "2nd"),
                  (scheme = "LCG", dt = 0.8,  h = 0.48,  cfl = 0.85,    space = "2nd", time = "2nd"),
                  (scheme = "FEM", dt = 0.1,  h = 1.20,  cfl = nothing, space = "2nd", time = "1st"),
                  (scheme = "FVM", dt = 3.31, h = 2.00,  cfl = 0.9,     space = "2nd", time = "2nd"),
                  (scheme = "McC", dt = 0.5,  h = 0.71,  cfl = 0.35,    space = "2nd", time = "2nd"),
                  (scheme = "STM", dt = 1.0,  h = 1.00,  cfl = nothing, space = "1st", time = "2nd")])

const ERROR_METRICS = ((:rms, "E_RMS"), (:max, "E_MAX"), (:sys, "E_SYS"), (:dias, "E_DIAS"))

# Table I for `case` plus the present scheme. `ours`: dt [ms], h [cm], cfl, space,
# time, polydeg, and optionally dt_cfl [ms], cfl_cfl for runs clipped by tstops.
function resolution_table(case::AbstractString, ours::NamedTuple)
    row = (; scheme = "present", dt = ours.dt, h = ours.h, cfl = ours.cfl,
           space = ours.space, time = ours.time)
    return TableSpec("Numerical parameters (Boileau et al. Table I, plus the present scheme)",
        [TableColumn(:scheme, "scheme"; fmt = :str),
         TableColumn(:dt, "dt [ms]"; fmt = :fixed, digits = 4),
         TableColumn(:h, "Omega_e [cm]"; fmt = :fixed, digits = 3),
         TableColumn(:cfl, "CFL"; fmt = :fixed, digits = 2),
         TableColumn(:space, "space"; fmt = :str),
         TableColumn(:time, "time"; fmt = :str)],
        NamedTuple[BOILEAU_TABLE_I[case]..., row];
        notes = ["CFL = lambda_max dt / Omega_e (Boileau's definition), blank = NA.",
                 "Present: mean accepted dt over the run, Omega_e the element length, " *
                 "lambda_max = max(|U| + c) in the vessel.",
                 "Present: Trixi's CFL number c sets dt = c Omega_e / ((N+1) lambda_max), " *
                 @sprintf("so c = 1 at N = %d is %.2f in this definition.",
                          ours.polydeg, 1 / (ours.polydeg + 1)),
                 haskey(ours, :dt_cfl) ?
                 "Present: steps land on the output grid (tstops); the CFL condition " *
                 @sprintf("alone gives a mean dt of %.3f ms (CFL %.2f).", ours.dt_cfl, ours.cfl_cfl) :
                 "Present: full CFL steps, resampled to the 1 ms output grid by cubic Hermite."])
end

"""
    compare_to_3d(case; ref, ours, codes, resolution, dcg_published, sensitivity)

Tables for a periodic case (`"carotid"` or `"aorta"`): the errors of all seven 1D
solutions against `ref` (3D), Table I with the present row, the published values
and recomputed minus published, and the phase sensitivity of the present errors
(`sensitivity`: lag in ms => our cycle with the 3D placed that much later).
"""
function compare_to_3d(case::AbstractString; ref, ours, codes, resolution, dcg_published,
                       sensitivity)
    methods = collect(BENCHMARK_METHODS)
    errs(x) = map((num, r, pw) -> boileau_errors(num, r; pointwise = pw),
                  x, ref, BENCHMARK_POINTWISE)
    E = Dict("ours" => errs(ours))
    for m in methods
        E[m] = errs(codes[m])
    end

    qlabel = (p = "P", Q = "Q", dr = "dr", dp = "dP")
    published = BOILEAU_PUBLISHED[case]

    rows = NamedTuple[]
    for (k, mlabel) in ERROR_METRICS, q in BENCHMARK_QUANTITIES
        vals = [Symbol(m) => E[m][q][k] for m in methods]
        push!(rows, (; metric = mlabel, quantity = qlabel[q], ours = E["ours"][q][k], vals...))
    end
    columns = [TableColumn(:metric, "error"; fmt = :str),
               TableColumn(:quantity, "quantity"; fmt = :str),
               TableColumn(:ours, "present"; fmt = :pct, digits = 2),
               (TableColumn(Symbol(m), m; fmt = :pct, digits = 2) for m in methods)...]
    headline = TableSpec("Midpoint errors vs the 3D reference, all seven 1D solutions",
                         columns, rows;
                         notes = ["Every column computed here from the supplied waveforms with one " *
                                  "routine (Boileau et al. Eqs. 10-13), at the 3D time points.",
                                  "Boileau et al.'s phase: 3D t corresponds to raw-inflow time " *
                                  "t_ini + $(BOILEAU_3D_OFFSET[case]) s + t (mod T); t_ini is the " *
                                  "supplement README's last-cycle start.",
                                  "P normalized pointwise (E_DIAS by min P_3D); Q, dr, dP by the 3D " *
                                  "maximum over the cycle. dr as supplied by each code (ours: r - r_d)."])

    Epub = copy(E)
    Epub["DCG"] = errs(dcg_published)
    pub_rows = NamedTuple[]
    diff_rows = NamedTuple[]
    for (k, mlabel) in ERROR_METRICS, q in BENCHMARK_QUANTITIES
        pub = getfield(published, k)[q]
        push!(pub_rows, (; metric = mlabel, quantity = qlabel[q],
                         (Symbol(m) => pub[i] for (i, m) in enumerate(BENCHMARK_METHODS))...))
        push!(diff_rows, (; metric = mlabel, quantity = qlabel[q],
                          (Symbol(m) => 100 * Epub[m][q][k] - pub[i]
                           for (i, m) in enumerate(BENCHMARK_METHODS))...))
    end
    code_cols(fmt) = [TableColumn(:metric, "error"; fmt = :str),
                      TableColumn(:quantity, "quantity"; fmt = :str),
                      (TableColumn(Symbol(m), m; fmt, digits = 2) for m in BENCHMARK_METHODS)...]
    pub_spec = TableSpec("Published values, Boileau et al. $(published.table) (per cent)",
                         code_cols(:fixed), pub_rows)
    diff_spec = TableSpec("Recomputed minus published (percentage points)",
                          code_cols(:fixed), diff_rows;
                          notes = ["Cross-check only. Nonzero entries mean the published column " *
                                   "is not what this routine gives on the supplied waveform.",
                                   "DCG sampled $(round(Int, 1e3 * DCG_ROW_LAG)) ms later than the " *
                                   "headline table, as the paper did (row-index extraction; " *
                                   "DCG's file starts at t = 0.001)."])

    lags = sort!(collect(keys(sensitivity)))
    Es = Dict(l => errs(sensitivity[l]) for l in lags)
    sens_rows = NamedTuple[]
    for (k, mlabel) in ERROR_METRICS[1:2], q in BENCHMARK_QUANTITIES
        push!(sens_rows, (; metric = mlabel, quantity = qlabel[q],
                          (Symbol("lag$i") => Es[l][q][k] for (i, l) in enumerate(lags))...))
    end
    sens_spec = TableSpec("Phase sensitivity of the present errors",
        [TableColumn(:metric, "error"; fmt = :str),
         TableColumn(:quantity, "quantity"; fmt = :str),
         (TableColumn(Symbol("lag$i"), @sprintf("%+d ms", l); fmt = :pct, digits = 2)
          for (i, l) in enumerate(lags))...],
        sens_rows;
        notes = ["3D placed the given number of ms later than the headline phase " *
                 "(0 ms = headline; -1 ms = t_ini + $(round(BOILEAU_3D_OFFSET[case] - 0.001, digits = 3)) s).",
                 "SYS/DIAS are phase-invariant and omitted."])

    return (errors = headline, resolution = resolution_table(case, resolution),
            published = pub_spec, published_diff = diff_spec, sensitivity = sens_spec)
end

# ---------------------------------------------------------------------------------
# Driver shared by the carotid and aorta benchmarks (Boileau et al. Sections 4.2-4.3)
# ---------------------------------------------------------------------------------

# Q_in(t) = scale (dc + Σ_m amp_m sin(2πm (t + shift)/T + phase_m)), the inflow.txt
# Fourier series of the supplement, evaluated at the shifted raw-inflow time.
function fourier_inflow(c)
    return function (t)
        tt = t + c.inflow_shift
        s = c.inflow_dc
        for m in eachindex(c.inflow_amp)
            s += c.inflow_amp[m] * sin(2 * m * pi * tt / c.T + c.inflow_phase[m])
        end
        return c.inflow_scale * s
    end
end

"""
    run_benchmark(case; refine = case.refine, cfl = 2.0, outdir = nothing)

Carotid or aorta benchmark in CGS units. Boileau's wall law references the
stiffness to the diastolic area A_d (p = p_d at A_d), while the vessel starts at
rest at the smaller unstressed area of their Eq. (16), where p = 0. The equations
therefore use A0 = A_d, nu = 0.5 (so that K_p/√A_d = β/A_d with β = (4/3)√π E h)
and the offset p_d, which makes `pressure` absolute; the Windkessel then has
p_out = 0 and p_C0 = 0.

The solution is marched for `case.ncycles` periods with N = 4 and RDPK3Sp510;
the last cycle at the midpoint is compared with the 3D reference. With `outdir`
the figures and `<name>_errors.md` are written there. Returns the comparison data.
"""
function run_benchmark(c; refine = c.refine, cfl = 2.0, outdir = nothing)
    alpha = (c.zeta + 2) / (c.zeta + 1)
    nu = 0.5
    beta_stiff = (4 / 3) * sqrt(pi) * c.E * c.h0 / c.A_d
    A0_init = (sqrt(c.A_d) - c.p_d / beta_stiff)^2
    equations = ArteryFlowEquations1D(c.rho, c.mu, alpha, c.A_d, c.h0, c.E, nu;
                                      p_ext = 0.0, p_d = c.p_d)
    p_mean = c.Q_mean * (c.R1 + c.R2)
    c_operating = wave_speed((sqrt(A0_init) + p_mean / beta_stiff)^2, equations)
    @info "$(c.name): A0_init = $(round(A0_init, digits = 5)) cm^2, " *
          "c(operating) = $(round(c_operating / 100, digits = 3)) m/s"

    bc_out = BoundaryConditionWindkessel3Element(c.R1, c.R2, c.C; p_out = 0.0, p_C0 = 0.0)
    boundary_conditions = (x_neg = BoundaryConditionInflowRiemann(fourier_inflow(c)),
                           x_pos = bc_out)
    solver = DGSEM(LobattoLegendreBasis(4), flux_es, VolumeIntegralFluxDifferencing(flux_ec))
    mesh = TreeMesh(0.0, c.L, initial_refinement_level = refine, periodicity = false)
    initial_condition(x, t, equations) = SVector(A0_init, zero(A0_init))
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        source_terms = source_terms_friction,
                                        boundary_conditions = boundary_conditions)

    tspan = (0.0, c.ncycles * c.T)
    ode = semidiscretize_windkessel(semi, bc_out, tspan; track_volume = true)
    wk = ode.p
    sol = solve(ode, RDPK3Sp510(); dt = 1e-4, adaptive = false, ode_default_options()...,
                callback = StepsizeCallback(cfl = cfl), save_everystep = true)
    vol = volume_balance(sol, wk)
    @info "Volume balance drift: $(round(vol.drift, sigdigits = 2)) cm^3 " *
          "($(round(vol.relative, sigdigits = 2)) of the inflow volume)"

    # States on the 1 ms grid, and the inlet/midpoint/outlet series.
    states = hermite_states(sol, windkessel_rhs!, wk,
                            collect(range(tspan[1], tspan[2], step = 0.001)))
    n = length(states.t)
    A_in = zeros(n); A_mid = zeros(n); A_out = zeros(n); Q_mid = zeros(n); lam = zeros(n)
    for (k, y) in enumerate(states.u)
        u_dg = dg_state(y, wk)
        A_in[k], A_mid[k], A_out[k] = evaluate_dg(u_dg, semi, [0.0, c.L / 2, c.L])
        Q_mid[k] = evaluate_dg(u_dg, semi, [c.L / 2]; var = 2)[1]
        u = Trixi.wrap_array(u_dg, semi)
        lam[k] = maximum(Trixi.max_abs_speeds(SVector(u[1, i, e], u[2, i, e]), equations)[1]
                         for i in axes(u, 2), e in axes(u, 3))
    end
    tt = states.t

    # Series in the supplement's units (kPa, ml/s, mm, kPa) on the raw-inflow time.
    p(A) = pressure(SVector(A, zero(A)), equations)
    r_d = sqrt(c.A_d / pi)
    ours_full = (p = p.(A_mid) ./ 1e4,
                 Q = Q_mid,
                 dr = (sqrt.(A_mid ./ pi) .- r_d) .* 10,
                 dp = (p.(A_in) .- p.(A_out)) ./ 1e4)
    t_raw = tt .+ c.inflow_shift

    ref_data = readdlm(joinpath(c.datadir, "3D.txt"))
    tref = ref_data[:, 1]
    ref = (p = ref_data[:, 2], Q = ref_data[:, 3], dr = ref_data[:, 4], dp = ref_data[:, 5])

    anchor = ref_anchor(c.T, c.t_ini, BOILEAU_3D_OFFSET[c.name])
    ours = cycle_at_ref_times(t_raw, ours_full, c.T, anchor, tref)
    ours_prev = cycle_at_ref_times(t_raw, ours_full, c.T, anchor, tref; cycles_back = 1)
    codes, bench = load_benchmark_cycles(c.datadir, c.T, anchor, tref)
    dcg_published = published_dcg_cycle(bench, c.T, anchor, tref)
    sensitivity = Dict(l => cycle_at_ref_times(t_raw, ours_full, c.T, anchor + 1e-3 * l, tref)
                       for l in (-1, 0, 1))

    t_lo = last_cycle_start(t_raw, c.T, anchor, tref[end]) - c.inflow_shift
    amp_rel(a, b) = maximum(abs.(a .- b)) / max(maximum(a) - minimum(a), eps())
    @info "Periodicity (last vs previous cycle): " *
          "p $(round(100 * amp_rel(ours.p, ours_prev.p), digits = 2))%, " *
          "Q $(round(100 * amp_rel(ours.Q, ours_prev.Q), digits = 2))% of amplitude"

    # Resolution as in Boileau et al. Table I: mean accepted dt, element length,
    # and CFL = λ_max dt / Omega_e with λ_max the largest |U| + c over the cycle.
    dt_mean = (tspan[2] - tspan[1]) / sol.stats.naccept
    h_elem = c.L / 2^refine
    lam_max = maximum(lam[k] for k in eachindex(tt)
                      if t_lo - 1e-9 <= tt[k] <= t_lo + tref[end] + 1e-9)
    resolution = (dt = 1e3 * dt_mean, h = h_elem, cfl = lam_max * dt_mean / h_elem,
                  space = "5th", time = "5th", polydeg = 4)
    tables = compare_to_3d(c.name; ref, ours, codes, resolution, dcg_published, sensitivity)

    if outdir !== nothing
        mkpath(outdir)
        save_benchmark_figures(c, outdir, tref, ours, ref, windkessel_history(sol, wk))
        emit_tables([tables.errors, tables.resolution, tables.published,
                     tables.published_diff, tables.sensitivity],
                    outdir, c.name * "_errors"; title = c.title)
    end
    return (; ours, ref, tref, tables, resolution, dt_mean, h_elem)
end

function save_benchmark_figures(c, outdir, tref, ours, ref, wk_history)
    thesis_plot_style!()
    col_1d = "#2F6FC0"
    col_3d = "#D6402E"
    common = (framestyle = :box, gridalpha = 0.25, lw = 2, legend = :best,
              xlabel = L"t" * "  [s]", size = (600, 450),
              left_margin = 8 * Plots.mm, bottom_margin = 5 * Plots.mm,
              right_margin = 4 * Plots.mm, top_margin = 3 * Plots.mm)

    pp = plot(tref, ours.p; color = col_1d, label = "1D", ylabel = L"p" * "  [kPa]", common...)
    pdp = plot(tref, ours.dp; color = col_1d, label = "1D",
               ylabel = L"\Delta p" * "  inlet-outlet  [kPa]", common..., legend = false)
    pQ = plot(tref, ours.Q; color = col_1d, label = "1D", ylabel = L"Q" * "  [ml/s]",
              common..., legend = false)
    pr = plot(tref, ours.dr; color = col_1d, label = "1D", ylabel = L"\Delta r" * "  [mm]",
              common..., legend = false)
    plot!(pp, tref, ref.p; color = col_3d, ls = :dash, lw = 1.4, label = "3D")
    plot!(pdp, tref, ref.dp; color = col_3d, ls = :dash, lw = 1.4, label = "3D")
    plot!(pQ, tref, ref.Q; color = col_3d, ls = :dash, lw = 1.4, label = "3D")
    plot!(pr, tref, ref.dr; color = col_3d, ls = :dash, lw = 1.4, label = "3D")
    for (plt, name) in ((pp, "midpoint_pressure"), (pdp, "inlet_outlet_pressure_difference"),
                        (pQ, "midpoint_flow_rate"), (pr, "radius_change_from_diastole"))
        thesis_savefig(plt, joinpath(outdir, "$(c.name)_$name.png"))
    end

    # Windkessel outlet: pressure p(A_b) and capacitor pressure p_C, and the coupling
    # flow Q_J. Both panels use the same tick format so their plot areas line up.
    d = wk_history
    tick_fmt = y -> @sprintf("%.1f", y)
    p1 = plot(d.t, d.p_out ./ 1e4; label = L"p(A_b)", color = "#D6402E", lw = 2,
              ylabel = L"p" * "  [kPa]", framestyle = :box, gridalpha = 0.25,
              yformatter = tick_fmt, left_margin = 8 * Plots.mm, top_margin = 4 * Plots.mm)
    plot!(p1, d.t, d.p_C ./ 1e4; label = L"p_C", color = "#2F6FC0", lw = 1.4, ls = :dash)
    p2 = plot(d.t, d.Q_out; label = L"Q_{\mathrm{J}}", color = "#2F9E63", lw = 2,
              xlabel = L"t" * "  [s]", ylabel = L"Q" * "  [ml/s]",
              c.windkessel_flow_axis(d.Q_out)...,
              framestyle = :box, gridalpha = 0.25, yformatter = tick_fmt,
              left_margin = 8 * Plots.mm, bottom_margin = 4 * Plots.mm)
    thesis_savefig(plot(p1, p2; layout = (2, 1), size = (1000, 700)),
                   joinpath(outdir, "$(c.name)_windkessel.png"))
    return nothing
end
