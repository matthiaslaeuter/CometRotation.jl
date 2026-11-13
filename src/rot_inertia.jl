struct Quadrature
    lambda::Vector{SVector{3,Float64}}
    w::Vector{Float64}
end

function Quadrature()
    d15 = sqrt(15.0)
    t1 = (6 + d15) / 21
    t2 = (9 - 2d15) / 21
    t3 = (6 - d15) / 21
    t4 = (9 + 2d15) / 21
    dxi  = (1/3, t1, t2, t1, t3, t4, t3)
    deta = (1/3, t1, t1, t2, t3, t3, t4)
    lambda = [let b0 = 1 - dxi[q] - deta[q], b1 = dxi[q]
                 SVector(1 - b0 - b1, b0, b1)
             end for q in 1:7]
     w1 = (155 + d15) / 1200
     w2 = (155 - d15) / 1200
     w = [270/1200, w1, w1, w1, w2, w2, w2]
    return Quadrature(lambda, w)
end

integrate(q::Quadrature, area::Real, values::AbstractVector) =
    area * sum(q.w[i] * values[i] for i in eachindex(values))

const quad7 = Quadrature()


function mesh_volume(m::SurfaceMesh)
    v = 0.0
    for f in 1:nfaces(m)
        n = m.normal[f]
        vals = [dot(n, bary2pos(m, f, lambda)) / 3 for lambda in quad7.lambda]
        v += integrate(quad7, m.area[f], vals)
    end
    return v
end


function mesh_centroid(m::SurfaceMesh, volume::Real)
    c = zero(Vec3)
    for f in 1:nfaces(m)
        n = m.normal[f]
        vals = [0.5 * dot(x, x) * n for x in (bary2pos(m, f, lambda) for lambda in quad7.lambda)]
        c += integrate(quad7, m.area[f], vals)
    end
    return c / volume
end


function inertia_kernel(n::Vec3, x::Vec3)::Mat3
    xx = dot(x, x)
    xn = dot(x, n)
    return @SMatrix [
        (-0.5*x[1]*n[1] + 0.3*xn)*xx   (-0.5*x[1]*n[2])*xx            (-0.5*x[1]*n[3])*xx
        (-0.5*x[2]*n[1])*xx            (-0.5*x[2]*n[2] + 0.3*xn)*xx   (-0.5*x[2]*n[3])*xx
        (-0.5*x[3]*n[1])*xx            (-0.5*x[3]*n[2])*xx            (-0.5*x[3]*n[3] + 0.3*xn)*xx
    ]
end

function moment_inertia(m::SurfaceMesh, x0::Vec3, rot::Mat3, volume::Real)::Mat3
    J = zero(Mat3)
    for f in 1:nfaces(m)
        rn = rot * m.normal[f]
        vals = [inertia_kernel(rn, rot * (x0 + bary2pos(m, f, lambda))) for lambda in quad7.lambda]
        J += integrate(quad7, m.area[f], vals)
    end
    rho = Constants.cg_mass / volume
    return rho * J
end

function principal_inertia(m::SurfaceMesh)::Mat3
    volume = mesh_volume(m)
    x0 = -mesh_centroid(m, volume)
    J1 = moment_inertia(m, x0, Mat3(I), volume)      
    _, vecs = principal_axes(J1)                      
    v2 = Vec3(vecs[:, 3])                             
    e1 = Vec3(1, 0, 0)
    v0 = e1 - dot(e1, v2) * v2                        
    v0 = v0 / norm(v0)
    v1 = cross(v2, v0)
    rot = Mat3(hcat(v0, v1, v2))                      
    return moment_inertia(m, x0, inv(rot), volume)
end
