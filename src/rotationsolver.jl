struct FitParams
    ra::Float64                
    dec::Float64               
    omega::Float64             
    weights::Vector{Float64}   
    free::Vector{Int}       
end

FitParams(ra, dec, omega, weights) =
    FitParams(ra, dec, omega, weights, collect(eachindex(weights)))

########################################

const reference_orbit  = rv_to_kepler(Constants.R0_SI, Constants.V0_SI; μ = Constants.GM_SUN_SI)
const reference_orbit  = rv_to_kepler(Constants.R0_SI, Constants.V0_SI; μ = Constants.GM_SUN_SI)
const reference_M      = true_to_mean_anomaly(reference_orbit.eccentricity,
                                              reference_orbit.anomaly)
const mean_motion      = sqrt(Constants.GM_SUN_SI / reference_orbit.semi_major_axis^3)

function sun_direction(jd::Real)

    t = (jd - Constants.reference_jd) * Constants.time2sec
    o = reference_orbit
    orbit_now = KeplerianElements{MeanAnomaly}(
        0.0, o.semi_major_axis, o.eccentricity, o.inclination,
        o.raan, o.argument_of_periapsis,
	mod2pi(reference_M + mean_motion * t))
    r, _ = kepler_to_rv(orbit_now; μ = Constants.GM_SUN_SI)
    return -Vec3(r) .* Constants.m2dist
end

########################################

struct Params_to_States
    emitter::Emitter
    tgrid::Vector{Float64}
    start::FitParams
end

function (statemap::Params_to_States)(p::FitParams)
 emit_w = with_weights(statemap.emitter, p.weights)
 grid = statemap.tgrid
 ic = setup2ic(emit_w.inertia, p.ra, p.dec, p.omega * Constants.time2sec)
 rhs = rotation_rhs(emit_w.inertia; torque = (t, R) -> torque(emit_w, sun_direction(t), R))
 prob = ODEProblem((u, p, t) -> rhs(t, u), ic, (grid[1], grid[end]))
 sol = solve(prob, RK4(); dt = grid[2] - grid[1], adaptive = false, saveat = grid)
 return sol.u
end

(statemap::Params_to_States)(beta::AbstractVector) = statemap(update_param(statemap.start, beta))

function update_param(p::FitParams, v::AbstractVector)
    w = copy(p.weights)
    w[p.free] = v[4:end]
    return FitParams(v[1], v[2], v[3] * Constants.sec2time, w, p.free)
end


struct Params_to_Observation
    statemap::Params_to_States
    obsgrid::Vector{Float64}
end

function (fitmap::Params_to_Observation)(beta::AbstractVector)
    sm = fitmap.statemap
    J_bf = sm.emitter.inertia
    grid = sm.tgrid
    states = sm(beta)
    n = length(fitmap.obsgrid)
    v = Vector{Float64}(undef, 3n)
    for (i, jd) in enumerate(fitmap.obsgrid)
        k = clamp(searchsortedlast(grid, jd), 1, length(grid) - 1)
        alpha = (jd - grid[k]) / (grid[k+1] - grid[k])
        y = (1 - alpha) .* states[k] .+ alpha .* states[k+1]
        omega = angular_velocity(J_bf, SVector(y[1], y[2], y[3], y[4]),
	                         Vec3(y[5], y[6], y[7]))
        r = norm(omega)
        v[3(i-1)+1] = atan(omega[2],omega[1]) 
        v[3(i-1)+2] = asin(omega[3]/r)
        v[3(i-1)+3] = r
    end
    return v
end


function time_grid()
    jd0 = Constants.day2jd(-400.0)
    jd1 = Constants.day2jd(400.0)

    dt = 1800.0 * Constants.sec2time
    seps = eps(Float32)           
    n = ceil(Int, (jd1 - seps - jd0) / dt) + 1
    grid = [jd0 + dt * (t - 1) for t in 1:n]
    grid[end] = jd1                           
    return grid
end
