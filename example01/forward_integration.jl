#!/usr/bin/env julia

using CometRotation

banner("Forward integration with CometRotation.jl")

section("Initialization")
emitter = Emitter("nodedef_67p.txt", "facedef_67p.txt")
param0  = FitParams(69.4, 64.0, 1.40699393389e-4, fill(1/8, 9))
tgrid   = time_grid()
obsdata = load_obsdata("Rosetta_data.txt", tgrid)
map_to_states = Params_to_States(emitter, tgrid, param0)
info("threads", Threads.nthreads())
info("faces", nfaces(emitter.mesh))
info("steps", length(tgrid))

section("Evolution")
t = @elapsed states = map_to_states(param0)
info("run time", secs(t))

section("Output")
statef = "forward_rotstates.txt"
write_states(statef, tgrid, states, emitter.inertia)
radecf, freqf = plot_states("forward", statef, obsdata)
info("state file", statef)
info("figures", radecf)
info("", freqf)

banner_end()
