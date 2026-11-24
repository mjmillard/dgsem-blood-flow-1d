# Regenerates every figure and table of the thesis into results/, running each
# experiment in its own Julia process (about 25 minutes on one core):
#
#     julia --project=. run_all.jl
const EXPERIMENTS = [
    ("convergence", ["experiments/convergence/convergence.jl"]),
    ("riemann", ["experiments/riemann/riemann.jl", "--T-sweep"]),
    ("single_pulse", ["experiments/single_pulse/single_pulse.jl"]),
    ("carotid", ["experiments/carotid/carotid.jl"]),
    ("aorta", ["experiments/aorta/aorta.jl"]),
]

cd(@__DIR__)
julia = `$(Base.julia_cmd()) --project=$(@__DIR__) --threads=1`
for (name, args) in EXPERIMENTS
    @info "Running $name"
    t = @elapsed run(`$julia $args`)
    @info "Finished $name in $(round(t, digits = 1)) s"
end
