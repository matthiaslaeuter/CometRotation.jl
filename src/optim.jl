
const FD_RELEPS = 10 * eps(Float32) 

function jacobian(formap, beta::AbstractVector, scale::AbstractVector)
    p = length(beta)
    f0 = formap(beta)
    J = Matrix{Float64}(undef, length(f0), p)
    Threads.@threads for i in 1:p
        h = max(abs(scale[i]) * FD_RELEPS, FD_RELEPS)
        gi = x -> (beta_i = Vector{Float64}(beta); beta_i[i] = x[1]; formap(beta_i))
        J[:, i] = FiniteDiff.finite_difference_jacobian(gi, [beta[i]], Val(:forward);
                                                        relstep = 0.0, absstep = h)
    end
    return J, f0
end

function newton_step(obsmap, fp::ObsVector, beta::AbstractVector, lambda::AbstractVector;
                       center = zero(beta), damping = (0.0,))
    J, model = jacobian(obsmap, beta, fp.scale)
    J = J ./ fp.sigma
    r = (model .- fp.data) ./ fp.sigma
    dev0 = residuum(fp,model)

    F = svd(J)
    U, Sigma, V = F.U, F.S, F.V
    rhs = Sigma .* (U' * r) .+ V' * (lambda .* (beta .- center))

    maxhalvings = 8
    alphas = 0.5 .^ (0:maxhalvings)
    for mu in damping
        d     = mu * dev0^2 ./ fp.scale .^ 2
        A     = V' * ((lambda .+ d) .* V) .+ Diagonal(Sigma .^ 2)
        delta = V * (A \ rhs)

        devs = similar(alphas)
        Threads.@threads for k in eachindex(alphas)
            devs[k] = residuum(fp, obsmap(beta .- alphas[k] .* delta))
        end

        k = findfirst(<(dev0), devs)
        k === nothing || return beta .- alphas[k] .* delta, devs[k]
    end
    return beta, dev0
end

function pushpositive(p::FitParams, dev::Real; upper::Real = Inf)
    lambda = zeros(nparams(p))
    for (i, w) in enumerate(p.weights[p.free])
        (w < 0 || w > upper) && (lambda[3+i] = dev^2 * 1.0e2)
    end
    return lambda
end

nparams(p::FitParams) = 3 + length(p.free)

free_weights(p::FitParams) = vcat(p.ra, p.dec, p.omega * Constants.time2sec, p.weights[p.free])
