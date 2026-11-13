struct SurfaceMesh
    nodes::Vector{Vec3}
    faces::Vector{NTuple{3,Int}}
    patch::Vector{Int}
    centroid::Vector{Vec3}
    normal::Vector{Vec3}
    area::Vector{Float64}
end

function SurfaceMesh(nodes::Vector{Vec3}, faces::Vector{NTuple{3,Int}}, patch::Vector{Int})
    n = length(faces)
    centroid = Vector{Vec3}(undef, n)
    normal   = Vector{Vec3}(undef, n)
    area     = Vector{Float64}(undef, n)
    for f in 1:n
        i, j, k = faces[f]
        a, b, c = nodes[i], nodes[j], nodes[k]
        centroid[f] = (a + b + c) / 3
        cx = cross(b - a, c - a)
        nrm = norm(cx)
        area[f]   = nrm / 2
        normal[f] = cx / nrm
    end
    return SurfaceMesh(nodes, faces, patch, centroid, normal, area)
end

nfaces(m::SurfaceMesh) = length(m.faces)

function bary2pos(m::SurfaceMesh, f::Int, lambda::SVector{3})
    i, j, k = m.faces[f]
    return lambda[1] * m.nodes[i] + lambda[2] * m.nodes[j] + lambda[3] * m.nodes[k]
end

function load_mesh(nodefile::AbstractString, facefile::AbstractString)
    nodes = Vec3[]
    for line in eachline(nodefile)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        p = split(s)
        push!(nodes, Vec3(parse(Float64, p[2]) * Constants.m2dist,
                          parse(Float64, p[3]) * Constants.m2dist,
                          parse(Float64, p[4]) * Constants.m2dist))
    end

    faces = NTuple{3,Int}[]
    patch = Int[]
    for line in eachline(facefile)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        p = split(s)
        push!(faces, (parse(Int, p[2]) + 1, parse(Int, p[3]) + 1, parse(Int, p[4]) + 1))
        push!(patch, parse(Int, p[5]))
    end

    return SurfaceMesh(nodes, faces, patch)
end

function sph2cart(lon::Real, lat::Real, r::Real = 1.0)::Vec3
    lambda = deg2rad(lon)
    phi = deg2rad(lat)
    return Vec3(r * cos(phi) * cos(lambda), r * cos(phi) * sin(lambda), r * sin(phi))
end

function cart2sph(x::Vec3)::Vec3
    r = norm(x)
    lon = rad2deg(atan(x[2], x[1]))
    lat = rad2deg(asin(x[3] / r))
    return Vec3(lon, lat, r)
end
