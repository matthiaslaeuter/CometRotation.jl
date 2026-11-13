const banner_width = 66

function banner(title::AbstractString)
    println()
    println("=" ^ banner_width)
    println("  ", title)
    println("=" ^ banner_width)
end

function banner_end()
    println("=" ^ banner_width)
    println()
end

function section(name::AbstractString)
    println()
    println("  ", name)
end

function info(label::AbstractString, value; indent::Int = 4, width::Int = 22)
    println(" " ^ indent, rpad(label, width), value)
end

fmt(x::Integer) = string(x)
fmt(x::AbstractFloat) =
    if x == 0
        "0"
    elseif abs(x) < 1e-3 || abs(x) >= 1e5
        string(round(x, sigdigits = 3))
    else
        string(round(x, digits = 6))
    end
fmt(x) = string(x)

secs(t::Real) = string(round(t, digits = 1), " s")

# --- figures ----------------------------------------------------------------

const jd_perihelion_day = Constants.jd_perihel_tk * Constants.time2day
const ra_scale = cosd(64.0)         

function read_states(path::AbstractString)
    rows = [parse.(Float64, split(l)) for l in eachline(path) if !isempty(strip(l))]
    days = Float64[]; ra = Float64[]; dec = Float64[]; omega = Float64[]
    for r in rows[1:8:end]
        lon, lat, mag = cart2sph(Vec3(r[5], r[6], r[7]))
        push!(days, r[1] - jd_perihelion_day)
        push!(ra, lon * ra_scale)
        push!(dec, lat)
        push!(omega, mag * Constants.sec2day)
    end
    return days, ra, dec, omega
end

function read_obs(obs::Observations)
    days  = obs.jd .* Constants.time2day .- jd_perihelion_day
    ra    = rad2deg.(obs.raw[:, 1]) .* ra_scale
    dec   = rad2deg.(obs.raw[:, 2])
    omega = obs.raw[:, 3] .* Constants.sec2time
    return days, ra, dec, omega
end

function plot_states(name::AbstractString, statefile::AbstractString, obs::Observations)
    curves = [(read_states(statefile)..., "numerical model",             :red),
              (read_obs(obs)...,        "Rosetta, Kramer et al. 2019", :black)]

    radec = Plots.plot(size = (700, 700), legend = :topright, tickfontsize = 12,
                       guidefontsize = 15, xlabel = "ra × cos(64°)", ylabel = "dec")
    freq  = Plots.plot(Size = (700, 400), legend = :bottomright, tickfontsize = 12,
                       guidefontsize = 15, xlabel = "time since perihelion (day)",
                       ylabel = "|ω| (s⁻¹)",
                       left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

    for (days, ra, dec, omega, label, color) in curves
        Plots.plot!(radec, ra, dec; label = label, color = color, lw = 2.5, legendfontsize=14)
        Plots.scatter!(radec, [linear_interpolation(days, ra)(0.0)], [linear_interpolation(days, dec)(0.0)];
                       color = color, ms = 6, label = "")
        Plots.plot!(freq, days, omega; label = label, color = color, lw = 2.5, legendfontsize=8)
    end

    radec_file, freq_file = "radec_$(name).pdf", "freq_$(name).pdf"
    Plots.savefig(radec, radec_file)
    Plots.savefig(freq, freq_file)
    return radec_file, freq_file
end



function write_states(path::AbstractString, grid, states, J_bf::Mat3)
    evals, evecs = principal_axes(J_bf)
    open(path, "w") do io
        for (jd, y) in zip(grid, states)
            row = rotstate_row(jd, y, J_bf, evals, evecs)
            println(io, join(row, " "))
        end
    end
end

function rotstate_row(jd::Real, y, J_bf::Mat3, evals, evecs)
    massunit = Constants.mass2kg * Constants.dist2m^2
    ounit = Constants.day2time
    lunit = massunit * ounit
    q = SVector(y[1], y[2], y[3], y[4])
    L = Vec3(y[5], y[6], y[7])
    R = Mat3(RotMatrix(QuatRotation(q[1], q[2], q[3], q[4])))
    omega = angular_velocity(J_bf, q, L)
    row = Float64[jd * Constants.time2day]
    append!(row, L .* lunit)
    append!(row, omega .* ounit)
    for i in 1:3
        axis_i = R * Vec3(evecs[:, i])
        append!(row, axis_i .* (evals[i] * massunit))
    end
    return row
end
