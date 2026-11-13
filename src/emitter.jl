struct Emitter
    mesh::SurfaceMesh
    inertia::Mat3       
    tpm::ThermalModel
    shadow::ShadowCache
    face_weight::Vector{Float64}  
end

expand_face_weights(mesh::SurfaceMesh, w::AbstractVector) =
    [Float64(w[mesh.patch[f] + 1]) for f in 1:nfaces(mesh)]

function Emitter(nodefile::AbstractString, facefile::AbstractString)
    mesh   = load_mesh(nodefile, facefile)
    npatch = maximum(mesh.patch) + 1
    return Emitter(mesh, principal_inertia(mesh), ThermalModel(), ShadowCache(mesh),
                   expand_face_weights(mesh, fill(1/8, npatch)))
end

function with_weights(em::Emitter, w::AbstractVector)
    return Emitter(em.mesh, em.inertia, em.tpm, em.shadow,
                   expand_face_weights(em.mesh, w))
end

function torque(em::Emitter, r2s::Vec3, R::Mat3)
    r2s_bf = transpose(R) * r2s                 
    r2s_unit = r2s_bf / norm(r2s_bf)
    r_au = norm(r2s_bf) * Constants.dist2au
    flux = Constants.solarconst_w_m2 * (1 / r_au)^2

    lo, la = shadow_bin(em.shadow, r2s_unit)

    bfmom = zero(Vec3)
    @inbounds for f in 1:nfaces(em.mesh)
        em.shadow.covered[f, lo, la] && continue
        ny = em.mesh.normal[f]
        q = flux * max(dot(r2s_unit, ny), 0.0)
        mdot, v = flux_velocity(em.tpm, q)          
        force = -(mdot * v) * ny
        da = em.face_weight[f] * em.mesh.area[f]
        bfmom += da * cross(em.mesh.centroid[f], force)
    end
    return R * bfmom                          
end
