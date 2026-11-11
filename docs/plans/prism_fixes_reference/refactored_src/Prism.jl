module Prism

using LinearAlgebra
using SparseArrays
using Statistics

include("loss.jl")
include("penalty.jl")
include("problem.jl")
include("result.jl")
include("optimality.jl")
include("solvers/fista_core.jl")
include("solvers/fista.jl")
include("solvers/group_fista.jl")
include("solvers/admm.jl")
include("workflows/paths.jl")
include("api.jl")
include("workflows/cross_validation.jl")
include("solvers/constrained.jl")
include("solvers/block_fista.jl")
include("solvers/laplacian_fista.jl")
include("solvers/tv_pdhg.jl")
include("compatibility_api.jl")

export GroupLassoProblem, GroupFISTASolver, group_lasso_kkt_residual
export group_lasso_dual_gap
export DataLoss, LeastSquares, Huber, L1
export LassoProblem, KKTStop, GradientMappingStop, DualGapStop, SolveResult
export FISTASolver, ADMMSolver, solve!, solve_path!, lasso_kkt_residual
export objective_value, lasso_dual_gap
export lasso_fista, nonnegative_lasso_fista, lasso_admm, nonnegative_lasso_admm
export LassoFISTASolver, LassoADMMSolver

end
