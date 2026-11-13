module CometRotation

using LinearAlgebra
using StaticArrays
using OrdinaryDiffEqLowOrderRK
using Rotations: QuatRotation, RotMatrix, params
using SatelliteToolboxBase: rv_to_kepler, kepler_to_rv, true_to_mean_anomaly,
                            KeplerianElements, MeanAnomaly
using Interpolations: linear_interpolation
import FiniteDiff
import Plots

include("constants.jl")
include("emit_thermo.jl")
include("emit_surface.jl")
include("emit_shadow.jl")
include("emitter.jl")
include("rot_ic.jl")
include("rot_inertia.jl")
include("rot_dynamics.jl")
include("rotationsolver.jl")
include("observation.jl")
include("optim.jl")
include("output.jl")

export Constants
export Emitter, nfaces
export FitParams, nparams, free_weights, update_param, pushpositive
export time_grid, Params_to_States, Params_to_Observation
export load_obsdata, create_ObsVector, residuum
export newton_step
export write_states, plot_states
export banner, banner_end, section, info, fmt, secs

end 
