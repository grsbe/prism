"""
    LassoProblem(A, b; constraint=:signed, loss=LeastSquares(), penalty=L1())

The algebraic problem `min_x ℓ(A*x-b) + λ*R(x)`, with the default
least-squares loss `ℓ(r) = 0.5*norm(r)^2` and L1 penalty `R(x) = norm(x,1)`.
`loss=Huber(delta)` selects the robust Huber loss and `penalty=Huber(delta)`
the (non-sparse) Huber penalty; both are FISTA only.
`constraint=:nonnegative` adds `x ≥ 0`. Background subtraction, physical
dictionaries, and whitening belong in the construction of `A` and `b`.
"""
struct LassoProblem{MA<:AbstractMatrix,VB<:AbstractVector,L<:DataLoss,
                    R<:CoefficientPenalty}
    A::MA
    b::VB
    constraint::Symbol
    loss::L
    penalty::R
end

function LassoProblem(A::AbstractMatrix, b::AbstractVector; constraint=:signed,
                      loss=LeastSquares(), penalty=L1())
    size(A, 1) == length(b) ||
        throw(DimensionMismatch("A rows must match length(b)"))
    size(A, 2) > 0 || throw(ArgumentError("A must have at least one column"))
    length(b) > 0 || throw(ArgumentError("b must be nonempty"))
    eltype(A) <: AbstractFloat && eltype(b) <: AbstractFloat ||
        throw(ArgumentError("A and b must use floating-point elements"))
    eltype(A) == eltype(b) ||
        throw(ArgumentError("A and b must have the same element type"))
    constraint in (:signed, :nonnegative) ||
        throw(ArgumentError("constraint must be :signed or :nonnegative"))
    finite_A = A isa AbstractSparseMatrix ? all(isfinite, nonzeros(A)) :
               all(isfinite, A)
    finite_A && all(isfinite, b) ||
        throw(ArgumentError("A and b must contain only finite values"))
    return LassoProblem(A, b, constraint, _convert_loss(loss, eltype(b)),
                        _convert_penalty(penalty, eltype(b)))
end

function _validate_lambda(lambda)
    lambda isa Real && isfinite(lambda) && lambda >= 0 ||
        throw(ArgumentError("lambda must be a finite nonnegative real number"))
    return lambda
end

function _validate_stop_number(x, name)
    x isa Real && isfinite(x) && x >= 0 ||
        throw(ArgumentError("$name must be a finite nonnegative real number"))
    return x
end
