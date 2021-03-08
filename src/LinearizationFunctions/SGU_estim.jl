@doc raw"""
    SGU_estim(XSS,A,B,m_par,n_par,indexes_aggr,distrSS;estim)

Calculate the linearized solution to the non-linear difference equations defined
by function [`Fsys`](@ref), while only differentiating with respect to the
aggregate part of the model, [`Fsys_agg()`](@ref).

The partials of the Jacobian belonging to the heterogeneous agent part of the model
are taken from the full-model derivatives provided as arguments, `A` and `B` (computed
by [`SGU()`](@ref)).

# Arguments
- `XSS`: steady state around which the system is linearized
- `A`,`B`: derivative of [`Fsys()`](@ref) with respect to arguments `X` [`B`] and
    `XPrime` [`A`]
- `m_par::ModelParameters`, `n_par::NumericalParameters`: `n_par.sol_algo` determines
    the solution algorithm
- `indexes::IndexStruct`,`indexes_aggr::IndexStructAggr`: access aggregate states and controls by name
- `distrSS::Array{Float64,3}`: steady state joint distribution

# Returns
as in [`SGU()`](@ref)
"""
function SGU_estim(XSSaggr::Array, A::Array, B::Array,
    m_par, n_par::NumericalParameters, indexes,
    indexes_aggr, distrSS::AbstractArray; estim=estim,Fsys_agg::Function = Fsys_agg)

    ############################################################################
    # Prepare elements used for uncompression
    ############################################################################

    ############################################################################
    # Check whether Steady state solves the difference equation
    ############################################################################

    length_X0 = length(XSSaggr) # Convention is that profits is the last control
    Bd = zeros(length_X0, length_X0)
    Ad = zeros(length_X0, length_X0)

    X0 = zeros(length_X0) .+ ForwardDiff.Dual(0.0,tuple(zeros(n_FD)...))

    F  = Fsys_agg(X0,X0,XSSaggr,distrSS,m_par,n_par,indexes_aggr)

    @make_deriv_estim n_FD
    BLAS.set_num_threads(12)

    prime_loop_estim!(Ad, DerivPrime, length_X0, n_par,n_FD)
    loop_estim!(Bd, Deriv, length_X0, n_par,n_FD)

    for k = 1:length(n_par.aggr_names)
        if !(any(distr_names.==n_par.aggr_names[k]))
            j = getfield(indexes, Symbol(n_par.aggr_names[k]))
            for h = 1:length(n_par.aggr_names)
                if !(any(distr_names.==n_par.aggr_names[h]))
                    i = getfield(indexes, Symbol(n_par.aggr_names[h]))
                    A[j,i] = Ad[k,h]
                    B[j,i] = Bd[k,h]
                end
            end
        end
    end
    #A[n_par.Asel] = Ad
    #B[n_par.Bsel] = Bd

    ############################################################################
    # Solve the linearized model: Policy Functions and LOMs
    ############################################################################
    BLAS.set_num_threads(Threads.nthreads())
    gx,hx, alarm_sgu,nk = SolveDiffEq(A,B,n_par;estim=estim)
    return gx, hx, alarm_sgu, nk, A, B
end

# Calculating Jacobian to XPrime
function prime_loop_estim!(Ad, DerivPrime, length_X0, n_par, n_FD)
    Threads.@threads for i = 1:n_FD:length_X0#in eachcol(A)
        aux = DerivPrime.(i)
        for k = 1:min(n_FD,length_X0-i+1)
            for j = 1:size(Ad,1)
                Ad[j,i+k-1] = aux[j][k]
            end
        end
    end
end

# Calculating Jacobian to X
function loop_estim!(Bd, Deriv, length_X0, n_par, n_FD)
    Threads.@threads for i = 1:n_FD:length_X0#in eachcol(A)
        aux = Deriv.(i)
        for k = 1:min(n_FD,length_X0-i+1)
            for j = 1:size(Bd,1)
                Bd[j,i+k-1] = aux[j][k]
            end
        end
    end
end
