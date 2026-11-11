# Records the exact behaviour of every FISTA-based Prism solver on a fixed set
# of problems. Usage:
#   julia --project=<prism checkout> fista_golden.jl <output file>
# Compare two recordings with fista_golden_compare.jl.
using Prism, LinearAlgebra, SparseArrays, Random, Serialization, Logging

record = Dict{String,Any}()
function keep!(name, result)
    fields = Dict{Symbol,Any}()
    for f in fieldnames(typeof(result))
        v = getfield(result, f)
        fields[f] = v isa AbstractArray ? collect(v) : v
    end
    haskey(record, name) && error("duplicate case name $name")
    record[name] = fields
end

stops = Dict(
    "gap" => DualGapStop(maxiter=3_000),
    "gap5" => DualGapStop(tol=1e-6, check_every=5, maxiter=3_000),
    "kkt" => KKTStop(rtol=1e-5, check_every=7, maxiter=3_000),
    "map" => GradientMappingStop(rtol=1e-5, maxiter=3_000),
    "short" => DualGapStop(tol=1e-12, maxiter=40),
)

with_logger(NullLogger()) do
    for T in (Float64, Float32)
        rng = MersenneTwister(42)
        m, n = 30, 48
        A = randn(rng, T, m, n)
        truth = zeros(T, n); truth[[3, 17, 30, 41]] = T[2, -1.5, 1, 3]
        b = A * truth + T(0.05) * randn(rng, T, m)
        b_outlier = copy(b); b_outlier[5] += 25
        B = randn(rng, T, m, 3)
        lambdas = T[3, 0.5, 0]
        for (sname, stop) in stops, constraint in (:signed, :nonnegative), step in (:exact, :power)
            tag = "$T/$sname/$constraint/$step"
            method = FISTA(; step, stop)
            # plain LASSO, both losses and penalties, warm-started path
            for (lname, loss) in (("ls", LeastSquares()), ("huber", Huber(1.0))),
                (pname, penalty) in (("l1", L1()), ("hpen", Huber(0.5)))
                rhs = lname == "huber" ? b_outlier : b
                solver = initialize(LassoProblem(A, rhs; constraint, loss, penalty), method)
                for (k, lambda) in enumerate(lambdas)
                    keep!("lasso/$tag/$lname/$pname/$k", solve!(solver, lambda))
                end
                keep!("lasso/$tag/$lname/$pname/cold", solve!(solver, lambdas[2]; warm_start=false))
            end
            # sparse matrix
            keep!("sparse/$tag", solve(LassoProblem(sparse(A), b; constraint), FISTA(; step=:power, stop), lambdas[2]))
            # group LASSO
            groups = [collect(i:i+3) for i in 1:4:n]
            for (lname, loss) in (("ls", LeastSquares()), ("huber", Huber(1.0)))
                solver = initialize(GroupLassoProblem(A, lname == "huber" ? b_outlier : b, groups; constraint, loss), method)
                for (k, lambda) in enumerate(lambdas)
                    keep!("group/$tag/$lname/$k", solve!(solver, lambda))
                end
            end
            # block: dense and matrix-free, ridge positive and zero
            for (dname, AA, BB) in (("dense", A, B), ("free", sparse(A), sparse(B))), ridge in T[0.7, 0]
                dname == "free" && step == :exact && continue
                for (pname, penalty) in (("l1", L1()), ("hpen", Huber(0.5)))
                    problem = BlockLassoRidgeProblem(AA, BB, b; x_constraint=constraint, z_constraint=:signed, penalty, loss=Huber(2.0))
                    solver = initialize(problem, method, ridge)
                    for (k, lambda) in enumerate(lambdas)
                        keep!("block/$tag/$dname/$ridge/$pname/$k", solve!(solver, lambda))
                    end
                end
                problem = BlockLassoRidgeProblem(AA, BB, b; x_constraint=:signed, z_constraint=:nonnegative)
                keep!("blockz/$tag/$dname/$ridge", solve(problem, method, lambdas[2], ridge))
            end
            # graph-Laplacian smoothing
            for smooth in T[0.3, 0]
                solver = initialize(LaplacianLassoProblem(A, b, (6, 8), smooth; constraint), method)
                for (k, lambda) in enumerate(lambdas)
                    keep!("laplacian/$tag/$smooth/$k", solve!(solver, lambda))
                end
            end
            # L1-ball constrained
            for radius in T[0, 2, 1000]
                solver = initialize(ConstrainedLassoProblem(A, b, radius; constraint), method)
                keep!("constrained/$tag/$radius/1", solve!(solver))
                keep!("constrained/$tag/$radius/2", solve!(solver))
                keep!("constrained/$tag/$radius/cold", solve!(solver; warm_start=false))
            end
        end
        # compatibility entry points
        keep!("compat/$T/a", lasso_fista(A, b, T(0.5); return_info=true))
        keep!("compat/$T/b", nonnegative_lasso_fista(A, b, T(0.5); reltol=1e-5, return_info=true))
        solver = LassoFISTASolver(A, b)
        keep!("compat/$T/c", lasso_fista(solver, T(0.5); return_info=true))
        keep!("compat/$T/d", nonnegative_lasso_fista(solver, T(0.2); warm_start=true, reltol=1e-4, return_info=true))
    end
end
serialize(ARGS[1], record)
println("recorded ", length(record), " cases to ", ARGS[1])
