@doc raw"""
    SGU(XSS,A,B,m_par,n_par,indexes,Copula,compressionIndexes,distrSS;estim)

Calculate the linearized solution to the non-linear difference equations defined
by function [`Fsys()`](@ref), using Schmitt-Grohé & Uribe (JEDC 2004) style linearization
(apply the implicit function theorem to obtain linear observation and
state transition equations).

The Jacobian is calculated using dual numbers (implemented by package `ForwardDiff`).
Use macro `@make_deriv` to compute partials simultaneously, with chunk size
given by `global` `n_FD`. Make use of model knowledge to set some entries manually.

# Arguments
- `XSS`: steady state around which the system is linearized
- `A`,`B`: matrices to be filled with first derivatives (see `Returns`)
- `m_par::ModelParameters`, `n_par::NumericalParameters`: `n_par.sol_algo` determines
    the solution algorithm
- `Copula::Function`,`distrSS::Array{Float64,3}`: `Copula` maps marginals to
    linearized approximation of joint distribution around `distrSS`
- `indexes::IndexStruct`,`compressionIndexes`: access states and controls by name
    (DCT coefficients of compressed ``V_m`` and ``V_k`` in case of
    `compressionIndexes`)

# Returns
- `gx`,`hx`: observation equations [`gx`] and state transition equations [`hx`]
- `alarm_sgu`,`nk`: `alarm_sgu=true` when solving algorithm fails, `nk` number of
    predetermined variables
- `A`,`B`: first derivatives of [`Fsys()`](@ref) with respect to arguments `X` [`B`] and
    `XPrime` [`A`]
"""
function SGU(XSS::Array,A::Array,B::Array, m_par, n_par::NumericalParameters,
    indexes, Copula::Function, compressionIndexes::Array{Int,1}, distrSS::Array{Float64,2}; estim=false,Fsys_agg::Function = Fsys_agg,balanced_budget=false)
    ############################################################################
    # Prepare elements used for uncompression
    ############################################################################
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ  = shuffleMatrix(distrSS, n_par)
    # Matrices for discrete cosine transforms
    DC1  = mydctmx(n_par.nk)
    DC2  = mydctmx(n_par.ny)

    ############################################################################
    # Check whether Steady state solves the difference equation
    ############################################################################
    # Convention is that profits is the last control
    length_X0 = n_par.ntotal+1
    # H as folded tensor
    H   = spzeros(length_X0,4*(length_X0)^2)
    F3  = zeros(length_X0,n_par.nstates+1)
    F1 = zeros(length_X0,n_par.nstates+1)
    F4  = zeros(length_X0,n_par.ncontrols)
    F2 = zeros(length_X0,n_par.ncontrols)
    # Set indexes where whole column of jacobian is constant
    @set! n_par.indexes_const = [indexes.Vk]
    @set! n_par.nstates_red = n_par.nstates
    @set! n_par.ncontrols_red = n_par.naggrcontrols
    @set! n_par.indexes_constP = [indexes.distr_k;indexes.distr_y]
    @set! n_par.nstates_redP = n_par.naggrstates
    @set! n_par.ncontrols_redP = n_par.ncontrols
    # Differentiate
    # Due to DuckTyping, Fsys_wrap calls Fsys for one asset (b/c compressionIndexes is not nested array)
    F(x,xp) = Fsys_wrap(x,xp,XSS,m_par,n_par,indexes,Γ,compressionIndexes,DC1,DC2,Copula;Fsys_agg=Fsys_agg,balanced_budget=balanced_budget)
    #BLAS.set_num_threads(1)
    builtin_FO_SO!(F3,F1,F4,F2,H,F,n_par;chunksize=19);
    # Trim FO derivatives by deleting row/column of permutation parameter
    F3  = F3[1:end-1,1:n_par.nstates]
    F4  = F4[1:end-1,:]
    F1 = F1[1:end-1,1:n_par.nstates]
    F2 = F2[1:end-1,:]
    # Build Bd,Ad from blocks
    B  = [F3 F4]
    A  = [F1 F2]
    # Redefine length_X0 for following calculations
    length_X0 = n_par.ntotal

    ############################################################################
    # Calculate Jacobians of the Difference equation F
    ############################################################################

    # Make use of the fact that Vk/Vm has no influence on any variable in
    # the system, thus derivative is 1
    B[[LinearIndices(B)[i,i] for i in n_par.indexes_const]] .= 1.0

    A[vcat([LinearIndices(A)[indexes.distr_k,i] for i in indexes.distr_k]...)] = reshape(-Γ[2][1:end-1,:],(:,1))
    
    A[vcat([LinearIndices(A)[indexes.distr_y,i] for i in indexes.distr_y]...)] = reshape(-Γ[3][1:end-1,:],(:,1))

    ############################################################################
    # Solve the linearized model: Policy Functions and LOMs
    ############################################################################
    BLAS.set_num_threads(Threads.nthreads())
    # BLAS.set_num_threads(1)
    gx, hx, alarm_sgu, nk = SolveDiffEq(A,B, n_par)
    return gx, hx, alarm_sgu, nk, A, B
end