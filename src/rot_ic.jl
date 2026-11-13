sym_axes(J::Mat3) = Mat3(transpose(principal_axes(J)[2]))

function align_axes(ref::Mat3, axis::Mat3)
    rows = Vector{Vec3}(undef, 3)
    for i in 1:3
        target = Vec3(ref[i, :])
        best, bestproj = 1, -1.0
        for j in 1:3
            p = abs(dot(target, Vec3(axis[j, :])))
            p > bestproj && ((best, bestproj) = (j, p))
        end
        v = Vec3(axis[best, :])
        rows[i] = dot(target, v) >= 0 ? v : -v
    end
    return Mat3(transpose(hcat(rows...)))
end

function omega0(scal_omega::Real, J_bf::Mat3, rm::Mat3)
    massis = inertia_inertial(J_bf, rm)
    axis = align_axes(Mat3(transpose(rm)), sym_axes(massis))
    b3 = Vec3(axis[3, :])
    return scal_omega * (b3 / norm(b3))
end

function orthomatrix(vec2::Vec3)
    e = (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))
    vec1 = e[1]
    for i in 1:3
        vec1 = e[i]
        abs(1 - abs(dot(vec1, vec2))) > eps(Float32) && break
    end
    vec1 = vec1 - dot(vec1, vec2) * vec2
    vec1 = vec1 / norm(vec1)
    vec0 = cross(vec1, vec2)
    return Mat3(hcat(vec0, vec1, vec2))
end

function setup2ic(J_bf::Mat3, ra::Real, dec::Real, spin::Real)
    rm = orthomatrix(sph2cart(ra, dec))
    q = SVector{4}(params(QuatRotation(RotMatrix{3}(rm))))
    omega = omega0(spin, J_bf, rm)
    L = inertia_inertial(J_bf, rm) * omega
    return SVector(q[1], q[2], q[3], q[4], L[1], L[2], L[3])
end
