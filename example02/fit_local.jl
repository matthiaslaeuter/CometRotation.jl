#!/usr/bin/env julia

using CometRotation

banner("Local fit with CometRotation.jl")

section("Initialization")
emitter       = Emitter("nodedef_67p.txt", "facedef_67p.txt")
param0        = FitParams(69.2721, 63.9857, 1.40444e-4,
              	         [0, 0.0368588511965, 0.00640770211497, 0.0203085839526, 0,
                             0.0159969812969, 0.00413885430986, 0.0748318857697, 0],
			 [2, 3, 4, 6, 7, 8])
tgrid         = time_grid()
obsdata       = load_obsdata("Rosetta_data.txt", tgrid)
map_to_states = Params_to_States(emitter, tgrid, param0)
map_to_obs    = Params_to_Observation(map_to_states, obsdata.jd)
info("parameters", nparams(param0))
info("threads", Threads.nthreads())

section("Newton method")
beta   = free_weights(param0)
obsobj = create_ObsVector(obsdata, beta)



dev    = residuum(obsobj, map_to_obs(beta))
info("start residual", fmt(dev))
t = @elapsed for it in 1:10
    global beta, dev
    prev = dev
    lambda = pushpositive(update_param(param0, beta), dev)
    beta, dev = newton_step(map_to_obs, obsobj, beta, lambda)
    info("step $it residual", "$(fmt(dev))")
    ( abs(prev - dev) < 1e-4 * prev) && break
end
info("run time", secs(t))

section("Report")
param  = update_param(param0, beta)
states = map_to_states(param)
statef = "local_fit_rotstates.txt"
write_states(statef, tgrid, states, emitter.inertia)
info("final residual", fmt(dev))
info("ra, dec", "$(fmt(param.ra)), $(fmt(param.dec))")
info("omega", fmt(param.omega))
info("weights", join(round.(param.weights; digits = 6), "  "))
info("positiveness", all(param.weights .>= 0) ? "yes" : "no")
info("state file",statef )

section("Output")
radecf, freqf = plot_states("optimized", statef, obsdata)
info("figures", radecf)
info("", freqf)

banner_end()
