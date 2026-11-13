using CometRotation
using CometRotation: Constants, Vec3, Mat3, inertia_inertial, angular_velocity,
                     rotation_rhs, sph2cart, cart2sph, SurfaceMesh, bary2pos, mesh_volume,
                     mesh_centroid, principal_inertia, ThermalModel, flux_velocity,
                     triangle_ray_hit, ShadowCache, torque, with_weights, sun_direction,
                     setup2ic
using LinearAlgebra
using StaticArrays
using OrdinaryDiffEqLowOrderRK
using Rotations: isrotation
using Test

const C = Constants

# helper: closed axis-aligned box mesh [0,lx]×[0,ly]×[0,lz] with outward normals
function box_mesh(lx, ly, lz)
    v = [Vec3(0, 0, 0), Vec3(lx, 0, 0), Vec3(lx, ly, 0), Vec3(0, ly, 0),
         Vec3(0, 0, lz), Vec3(lx, 0, lz), Vec3(lx, ly, lz), Vec3(0, ly, lz)]
    quads = [(1,2,3,4), (5,6,7,8), (1,2,6,5), (4,3,7,8), (1,4,8,5), (2,3,7,6)]
    tris = NTuple{3,Int}[]
    for (a, b, c, d) in quads
        push!(tris, (a, b, c)); push!(tris, (a, c, d))
    end
    center = Vec3(lx/2, ly/2, lz/2)
    oriented = map(tris) do (i, j, k)
        nrm = cross(v[j] - v[i], v[k] - v[i]); cen = (v[i] + v[j] + v[k]) / 3
        dot(nrm, cen - center) < 0 ? (i, k, j) : (i, j, k)
    end
    return SurfaceMesh(v, oriented, zeros(Int, length(oriented)))
end

@testset "CometRotation foundation" begin

    @testset "unit-system round trips" begin
        @test C.au2dist * C.dist2au ≈ 1
        @test C.m2dist  * C.dist2m  ≈ 1
        @test C.sec2time * C.time2sec ≈ 1
        @test C.day2time * C.time2day ≈ 1
        @test C.kg2mass  * C.mass2kg  ≈ 1
    end

    @testset "epochs" begin
        @test C.day2jd(-150.0) ≈ C.jd_perihel_tk - 150 * C.day2time
        @test C.day2jd(50.0)   ≈ C.jd_perihel_tk +  50 * C.day2time
    end

    @testset "residuum" begin
        # data [2, 4] with sigma [1, 2]
        obj = CometRotation.ObsVector([2.0, 4.0], [1.0, 2.0], [1.0, 1.0])
        @test residuum(obj, [2.0, 4.0]) ≈ 0                # model equals data
        @test residuum(obj, [4.0, 8.0]) ≈ sqrt(8)          # residuals (2, 4) / (1, 2) = (2, 2)
    end

    @testset "quaternion" begin
        q = normalize(@SVector [0.3, -0.5, 0.7, 0.2])
        J = @SMatrix [2.0 0 0; 0 3.0 0; 0 0 5.0]
        L = Vec3(0.4, -1.1, 2.3)

        # identity orientation: body frame = inertial frame, so omega = J \ L
        @test angular_velocity(J, SVector(1.0, 0, 0, 0), L) ≈ J \ L

        # a NON-unit quaternion must give the same omega: the RK4 stage vectors
        # have |q| ≈ 1.002, and an un-normalised rotation would corrupt J_is and omega
        for s in (1.002, 0.95, 1.3)
            @test angular_velocity(J, s .* q, L) ≈ angular_velocity(J, q, L) atol = 1e-12
        end

        # pin the quaternion convention (w, x, y, z) against the C++ entry formulas
        w, x, y, z = q
        Rc = @SMatrix [1-2(y^2+z^2)  2(x*y-w*z)   2(x*z+w*y)
                       2(x*y+w*z)    1-2(x^2+z^2) 2(y*z-w*x)
                       2(x*z-w*y)    2(y*z+w*x)   1-2(x^2+y^2)]
        @test angular_velocity(J, q, L) ≈ inertia_inertial(J, Rc) \ L atol = 1e-12

        qmul = CometRotation.qmul
        id = @SVector [1.0, 0, 0, 0]
        @test qmul(id, q) ≈ q
        @test qmul(q, id) ≈ q
        @test qmul(q, SVector(q[1], -q[2], -q[3], -q[4])) ≈ id atol = 1e-12   # q ⊗ conj(q)
    end

    @testset "dynamics: free rigid body" begin
        J = @SMatrix [2.0 0 0; 0 3.0 0; 0 0 5.0]      # triaxial inertia (Mat3)

        # Lowest-energy state for this J: spin about the MAX-inertia axis (e3).
        # Identity quaternion, so the body frame is the inertial frame and
        # L = J·ω = (0, 0, 5 × 1.5). Written out rather than built by a helper,
        # so the test states its own initial condition.
        spin = 1.5
        y0 = SVector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0 * spin)
        L0 = Vec3(y0[5], y0[6], y0[7])
        # lowest energy about the max-inertia axis -> L ∥ e3
        @test abs(dot(normalize(L0), Vec3(0, 0, 1))) ≈ 1 atol = 1e-10

        energy(y) = let q = SVector(y[1], y[2], y[3], y[4]), L = Vec3(y[5], y[6], y[7])
            0.5 * dot(angular_velocity(J, q, L), L)
        end
        Lof(y) = Vec3(y[5], y[6], y[7])

        # integrate torque-free with fixed-step RK4 through the SciML interface
        f = rotation_rhs(J; torque = (t, R) -> zero(Vec3))
        prob = ODEProblem((u, p, t) -> f(t, u), y0, (0.0, 20.0))
        sol = solve(prob, RK4(); dt = 0.1, adaptive = false)
        yend = sol.u[end]

        @test norm(Lof(yend)) ≈ norm(L0) atol = 1e-12   # torque-free: L conserved
        @test energy(yend) ≈ energy(y0) rtol = 1e-8     # steady rotation: energy conserved
        @test norm(SVector(yend[1], yend[2], yend[3], yend[4])) ≈ 1 rtol = 1e-4
    end

    @testset "time grid" begin
        grid = time_grid()
        @test grid[1] ≈ C.day2jd(-400.0)
        @test grid[end] ≈ C.day2jd(400.0)              # last point snapped to perihelion + 400 d
        @test issorted(grid)
        @test length(grid) == 38401                     # 800 days in 30-minute steps
        dt = 1800.0 * Constants.sec2time
        @test grid[2] - grid[1] ≈ dt                    # uniform interior step
        @test grid[end] - grid[end-1] <= dt + eps(Float32)  # last interval no longer than dt
    end

    @testset "quadrature" begin
        # reference triangle (0,0),(1,0),(0,1): ∫ x^i y^j dA = i! j! / (i+j+2)!
        # with x = λ2, y = λ3 for our node ordering.
        q = CometRotation.Quadrature()
        @test sum(q.w) ≈ 1
        exact(i, j) = factorial(i) * factorial(j) / factorial(i + j + 2)
        area = 0.5
        for (i, j) in ((2, 0), (0, 2), (1, 1), (3, 2), (2, 3), (5, 0))
            vals = [lambda[2]^i * lambda[3]^j for lambda in q.lambda]        # degree ≤ 5 -> exact
            @test CometRotation.integrate(q, area, vals) ≈ exact(i, j) atol = 1e-13
        end
    end

    @testset "mesh geometry" begin
        # single triangle in the z=0 plane
        nodes = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
        m = SurfaceMesh(nodes, [(1, 2, 3)], [0])
        @test nfaces(m) == 1
        @test m.area[1] ≈ 0.5
        @test m.normal[1] ≈ Vec3(0, 0, 1)
        @test m.centroid[1] ≈ Vec3(1/3, 1/3, 0)
        @test bary2pos(m, 1, SVector(1/3, 1/3, 1/3)) ≈ m.centroid[1]

        # reversed node winding flips the normal, area unchanged
        mr = SurfaceMesh(nodes, [(1, 3, 2)], [0])
        @test mr.normal[1] ≈ Vec3(0, 0, -1)
        @test mr.area[1] ≈ 0.5
        @test norm(m.normal[1]) ≈ 1               # normals are unit vectors
    end

    @testset "inertia tensor (TenInert)" begin
        M = Constants.cg_mass                        # box mass = cg_mass (any volume)
        m = box_mesh(3.0, 2.0, 1.0)
        @test mesh_volume(m) ≈ 6.0 atol = 1e-9
        @test mesh_centroid(m, mesh_volume(m)) ≈ Vec3(1.5, 1.0, 0.5) atol = 1e-9

        # uniform box principal moments: I = M(b²+c²)/12 per axis
        Ix = M * (2.0^2 + 1.0^2) / 12
        Iy = M * (3.0^2 + 1.0^2) / 12
        Iz = M * (3.0^2 + 2.0^2) / 12
        J = principal_inertia(m)
        @test J ≈ @SMatrix([Ix 0 0; 0 Iy 0; 0 0 Iz]) rtol = 1e-8
    end

    @testset "coordinates (System)" begin
        @test sph2cart(0.0, 0.0, 1.0) ≈ Vec3(1, 0, 0)
        @test sph2cart(90.0, 0.0, 1.0) ≈ Vec3(0, 1, 0) atol = 1e-12
        @test sph2cart(0.0, 90.0, 1.0) ≈ Vec3(0, 0, 1) atol = 1e-12
        @test cart2sph(Vec3(1, 0, 0)) ≈ Vec3(0, 0, 1)
        for v in (Vec3(0.3, -0.5, 0.8), Vec3(-1.0, 2.0, 0.5))
            ll = cart2sph(v)                       # round trip
            @test sph2cart(ll[1], ll[2], ll[3]) ≈ v
        end
    end

    @testset "thermophysics (model A TPM)" begin
        psc = C.molec_kg * C.kg2mass * C.time2sec * C.dist2m^2
        tm = ThermalModel()

        # midway between the 1000 and 2000 W/m² knots, the linear interpolant
        # must give the mean of the two tabulated production rates
        mdot, v = flux_velocity(tm, 1500.0)
        @test mdot ≈ 0.5 * (1.160942889242188e22 + 2.4219816868966826e22) * psc

        # both curves are positive and increase with irradiation
        mdot1, v1 = flux_velocity(tm, 500.0)
        mdot2, v2 = flux_velocity(tm, 1500.0)
        @test mdot1 > 0 && v1 > 0
        @test mdot2 > mdot1
        @test v2 >= v1

        @test flux_velocity(tm, 20.0)[1] ≈ 1.8426107804616492e16 * psc   # exactly at a knot
        @test_throws BoundsError flux_velocity(tm, 2500.0)               # outside the table
    end

    @testset "shadowing (triangle ray hit)" begin
        x0, x1, x2 = Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)
        @test triangle_ray_hit(Vec3(0.25, 0.25, 1), Vec3(0, 0, -1), x0, x1, x2)   # inside
        @test !triangle_ray_hit(Vec3(0.25, 0.25, 1), Vec3(0, 0, 1), x0, x1, x2)   # wrong way
        @test !triangle_ray_hit(Vec3(2, 2, 1), Vec3(0, 0, -1), x0, x1, x2)        # misses
    end

    @testset "emitter torque" begin
        m = box_mesh(2.0, 2.0, 2.0)
        R = Mat3(I)
        r2s = Vec3(2000.0, 0.0, 0.0)          # sun along +x, ~2 au (1 au = 1000 internal)

        # built field by field: the scripts' Emitter(nodefile, facefile) needs mesh files
        em1 = Emitter(m, principal_inertia(m), ThermalModel(), ShadowCache(m), ones(nfaces(m)))
        T = torque(em1, r2s, R)
        @test T isa Vec3 && all(isfinite, T)

        em0 = with_weights(em1, [0.0])
        @test norm(torque(em0, r2s, R)) ≈ 0 atol = 1e-14      # no activity -> no torque

        em2 = with_weights(em1, [2.0])
        @test torque(em2, r2s, R) ≈ 2 .* T                    # linear in the patch weight
    end

    @testset "initial state (setup2ic / orthomatrix)" begin
        b3 = normalize(Vec3(0.2, -0.5, 0.84))
        R = CometRotation.orthomatrix(b3)
        @test isrotation(R)
        @test R[:, 3] ≈ b3                       # spin axis is the third column

        # diagonal inertia -> lowest-energy IC with ω ∥ L ∥ spin axis
        J = @SMatrix [2.0 0 0; 0 3.0 0; 0 0 5.0]
        y = setup2ic(J, 69.4, 64.0, 0.5)
        q = SVector(y[1], y[2], y[3], y[4]); L = Vec3(y[5], y[6], y[7])
        axis = sph2cart(69.4, 64.0)
        @test norm(q) ≈ 1
        omega = angular_velocity(J, q, L)
        @test norm(omega) ≈ 0.5 atol = 1e-8                     # rate recovered
        @test abs(dot(normalize(omega), axis)) ≈ 1 atol = 1e-8  # ω ∥ spin axis
        @test abs(dot(normalize(L), axis)) ≈ 1 atol = 1e-8  # L ∥ spin axis
    end

    @testset "orbit: sun direction (regression)" begin
        # Pins sun_direction to reference values from an INDEPENDENT two-body
        # implementation (numpy, internal units). This is the test that makes it
        # safe to have removed the `gm` argument from the call chain: gm's only
        # consumer was sun_direction, so pinning sun_direction's output pins
        # everything gm could have affected.
        refs = Dict(                    # day offset from perihelion => position
            -400 => Vec3(-1.109447403280452e+03,  3.118848150455661e+03,  1.774568797577144e+03),
            -150 => Vec3(-1.812383506376837e+03,  8.271587704651156e+02,  6.331026373468081e+02),
               0 => Vec3(-5.673022602433778e+02, -1.001010164426566e+03, -4.707408582932730e+02),
              50 => Vec3( 3.575001074535311e+02, -1.164776130871089e+03, -6.566542554010252e+02),
             400 => Vec3( 3.552963257602954e+03,  1.191788550927721e+03,  2.525174461995849e+02),
        )
        for (days, ref) in refs
            d = sun_direction(C.jd_perihel_tk + days * C.day2time)
            @test d isa Vec3
            @test d ≈ ref rtol = 1e-10
        end
    end

    @testset "orbit: physics" begin
        # independent anchors, not reference values
        rp = norm(sun_direction(C.jd_perihel_tk)) * C.dist2au
        @test rp ≈ 1.24316136 atol = 1e-6                       # 67P perihelion

        # the orbit closes after one period
        a = CometRotation.reference_orbit.semi_major_axis
        T = 2pi * sqrt(a^3 / C.GM_SUN_SI)                        # seconds
        jd0 = C.jd_perihel_tk
        jd1 = jd0 + T / C.time2sec
        @test sun_direction(jd1) ≈ sun_direction(jd0) rtol = 1e-9

        # heliocentric distance stays between perihelion and aphelion
        e = CometRotation.reference_orbit.eccentricity
        for days in -400:50:400
            r = norm(sun_direction(jd0 + days * C.day2time)) * C.dist2m
            @test (1 - e) * a * 0.999 <= r <= (1 + e) * a * 1.001
        end
    end
end
