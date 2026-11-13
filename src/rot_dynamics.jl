inertia_inertial(J_bf::Mat3, R::Mat3)::Mat3 = R * J_bf * transpose(R)

function angular_velocity(J_bf::Mat3, q::SVector{4}, L::Vec3)::Vec3
    R = Mat3(RotMatrix(QuatRotation(q[1], q[2], q[3], q[4])))
    J_is = inertia_inertial(J_bf, R)
    return J_is \ L
end

quat_kinematics(omega::Vec3, q::SVector{4}) =
    0.5 .* qmul(SVector(0.0, omega[1], omega[2], omega[3]), q)

function rotation_rhs(J_bf::Mat3; torque)
    return function rhs(t, y)
        q = SVector(y[1], y[2], y[3], y[4])
        L = Vec3(y[5], y[6], y[7])
        R = Mat3(RotMatrix(QuatRotation(q[1], q[2], q[3], q[4])))
        omega = inertia_inertial(J_bf, R) \ L
        qdot = quat_kinematics(omega, q)
        T = torque(t, R)
        return SVector(qdot[1], qdot[2], qdot[3], qdot[4], T[1], T[2], T[3])
    end
end

function principal_axes(J_bf::Mat3)
    F = eigen(Symmetric(Matrix(J_bf)))
    return F.values, F.vectors
end

qmul(a::SVector{4}, b::SVector{4}) = SVector(
    a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4],
    a[1]*b[2] + a[2]*b[1] + a[3]*b[4] - a[4]*b[3],
    a[1]*b[3] - a[2]*b[4] + a[3]*b[1] + a[4]*b[2],
    a[1]*b[4] + a[2]*b[3] - a[3]*b[2] + a[4]*b[1])
