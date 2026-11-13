struct Observations
    jd::Vector{Float64}
    raw::Matrix{Float64}
    sigma::Matrix{Float64}
end

function load_obsdata(filename::AbstractString, tgrid::AbstractVector)
    jd = Float64[]
    rows = Vector{Float64}[]
    sigma_rows = Vector{Float64}[]
    for line in eachline(filename)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        p = parse.(Float64, split(s))
        push!(jd, p[1] * Constants.day2time)
        push!(rows,  [deg2rad(p[2]), deg2rad(p[3]), p[4] * Constants.time2day])
        push!(sigma_rows, [deg2rad(p[5]), deg2rad(p[6]), p[7] * Constants.time2day])
    end
    longobs = Observations(jd, permutedims(reduce(hcat, rows)), permutedims(reduce(hcat, sigma_rows)))
    lo, hi = extrema(tgrid)
    keep = findall(t -> lo <= t <= hi, longobs.jd)
    isempty(keep) && throw(ArgumentError("no observations inside the integration window"))
    shortobs = Observations(longobs.jd[keep], longobs.raw[keep, :], longobs.sigma[keep, :])
    return shortobs
end

observation_raw(obs::Observations) = vec(permutedims(obs.raw))
observation_sigma(obs::Observations)       = vec(permutedims(obs.sigma))

struct ObsVector
    data::Vector{Float64}
    sigma::Vector{Float64}
    scale::Vector{Float64}
end

residuum(ov::ObsVector, model) = norm( (model .- ov.data) ./ ov.sigma )

function create_ObsVector(obsdata, beta)
    rawvector = observation_raw(obsdata)
    sigmavector = observation_sigma(obsdata)
    return ObsVector(rawvector, sigmavector, abs.(beta))
end
