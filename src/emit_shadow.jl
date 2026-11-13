function triangle_ray_hit(g0::Vec3, dir::Vec3, x0::Vec3, x1::Vec3, x2::Vec3)
    tol = eps(Float64)
    a1 = x1 - x0; n1 = norm(a1); a1 = a1 / n1
    a2 = x2 - x0; n2 = norm(a2); a2 = a2 / n2
    nrm = cross(a1, a2)

    para = dot(nrm, dir)
    abs(para) <= tol && return false
    t = (dot(nrm, x0) - dot(nrm, g0)) / para
    t <= tol && return false

    p = g0 + t*dir - x0
    a12 = dot(a1, a2)
    det = 1 - a12^2
    r1 = dot(p, a1); r2 = dot(p, a2)
    alpha1 = (r1 - r2*a12) / (det * n1)
    alpha2 = (r2 - r1*a12) / (det * n2)
    return alpha1 > tol && alpha2 > tol && alpha1 + alpha2 < 1 - tol
end

function is_covered(m::SurfaceMesh, f::Int, dir::Vec3)
    dot(dir, m.normal[f]) < 0 && return true
    g0 = m.centroid[f]
    for g in 1:nfaces(m)
        g == f && continue
        i, j, k = m.faces[g]
        triangle_ray_hit(g0, dir, m.nodes[i], m.nodes[j], m.nodes[k]) && return true
    end
    return false
end

struct ShadowCache
    lonsize::Int
    latsize::Int
    covered::Array{Bool,3} 
end

function ShadowCache(m::SurfaceMesh, lonsize::Int = 40, latsize::Int = 40)
    dx = 360.0 / lonsize
    dy = 180.0 / latsize
    cov = Array{Bool,3}(undef, nfaces(m), lonsize, latsize)
    Threads.@threads for f in 1:nfaces(m)
        for lo in 1:lonsize
            lon = -180 + (lo - 1)*dx + 0.5dx
            for la in 1:latsize
                lat = -90 + (la - 1)*dy + 0.5dy
                cov[f, lo, la] = is_covered(m, f, sph2cart(lon, lat))
            end
        end
    end
    return ShadowCache(lonsize, latsize, cov)
end

function shadow_bin(sc::ShadowCache, dir::Vec3)
    ll = cart2sph(dir)
    lo = clamp(floor(Int, (ll[1] + 180) / (360.0 / sc.lonsize)) + 1, 1, sc.lonsize)
    la = clamp(floor(Int, (ll[2] +  90) / (180.0 / sc.latsize)) + 1, 1, sc.latsize)
    return lo, la
end


