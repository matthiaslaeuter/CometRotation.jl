#!/usr/bin/env julia

using Distributed

minutes = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 30.0
# nwork   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : max(1, div(Sys.CPU_THREADS, 2))

Threads.nthreads() == 1 && @warn "recommendation: start multithreaded with -t auto"
nwork = Threads.nthreads()
addprocs(nwork; exeflags = "--project=$(Base.active_project())")
@everywhere using CometRotation
using Metaheuristics


banner("Global fit with CometRotation.jl")

section("Initialization")
@everywhere begin
    emitter = Emitter("nodedef_67p.txt", "facedef_67p.txt")
    param0  = FitParams(69.4, 64.0, 1.40699393389e-4, fill(1/8, 9))
    tgrid   = time_grid()
    obsdata = load_obsdata("Rosetta_data.txt", tgrid)
    map_to_states = Params_to_States(emitter, tgrid, param0)
    map_to_obs    = Params_to_Observation(map_to_states, obsdata.jd)
    obsobj  = create_ObsVector(obsdata, free_weights(param0))
    spin0   = free_weights(param0)[1:3]  # ra, dec, omega
    residuum_observation(wts) = residuum(obsobj, map_to_obs(vcat(spin0, wts)))
end
residuum_observation_all(X) = pmap(residuum_observation, [X[i, :] for i in axes(X, 1)])
info("parameters", nparams(param0))
info("workers", nworkers())
info("threads", Threads.nthreads())
info("maximum search time", "$(minutes) min")


section("Global search with Metaheuristics")
n       = length(param0.free)
wmax    = 0.085
bounds  = boxconstraints(lb = zeros(n), ub = fill(wmax, n))
options = Options(time_limit = minutes * Constants.min2sec, iterations = 100_000,
                  f_calls_limit = 1e9, parallel_evaluation = true)
logger(st) = st.iteration % 5 == 0 &&
             info("generation $(st.iteration)", "residuum $(fmt(minimum(st)))   evals $(st.f_calls)")
meta_result = optimize(residuum_observation_all, bounds, DE(N = 96, F = 0.6, CR = 0.9, options = options);
               logger = logger)
weights_meta  = positions(meta_result)
residuum_meta  = fvals(meta_result)
twocandidates  = sortperm(residuum_meta)[1:2]
#for i in twocandidates
# info("residuum candidate", fmt(meta_result[i]))
#end
info("residuum meta", fmt(minimum(meta_result)))
info("evaluations", meta_result.f_calls)


section("Newton method")
boundcenter(x) = vcat(x[1:3], clamp.(x[4:end], 0.0, wmax))
function localnewton(w)
    beta  = vcat(spin0, w)
    res   = residuum(obsobj, map_to_obs(beta))
    steps = 0
    for it in 1:10
        prev   = res
        lambda = pushpositive(update_param(param0, beta), res; upper = wmax)
        beta, res = newton_step(map_to_obs, obsobj, beta, lambda;
                                  center = boundcenter(beta), damping = (0.0, 1e-2, 1.0, 1e2))
        res < prev || break
        steps += 1
        ( abs(prev - res) < 1e-4 * prev ) && break
    end
    return beta, res, steps
end
t = @elapsed fits = [localnewton(weights_meta[i, :]) for i in twocandidates]
for (i, (_, res, steps)) in zip(twocandidates, fits)
    info("residuum candidate", "$(fmt(residuum_meta[i]))  ->  $(fmt(res))   ($steps steps)")
end
info("run time", secs(t))
beta_best, res_best, _ = argmin(f -> f[2], fits)


section("Result")
param  = update_param(param0, beta_best)
states = map_to_states(param)
statef = "global_fit_rotstates.txt"
write_states(statef, tgrid, states, emitter.inertia)
info("residuum final", fmt(res_best))
info("ra, dec", "$(fmt(param.ra)), $(fmt(param.dec))")
info("omega", fmt(param.omega))
info("weights", join(round.(param.weights; digits = 6), "  "))
info("positiveness", all(param.weights .>= 0) ? "yes" : "no")
info("state file", statef)


section("Output")
radecf, freqf = plot_states("optimized", statef, obsdata)
info("figures", radecf)
info("", freqf)

banner_end()
