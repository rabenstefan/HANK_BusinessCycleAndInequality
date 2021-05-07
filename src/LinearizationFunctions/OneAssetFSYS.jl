@doc raw"""
    Fsys(X,XPrime,Xss,m_par,n_par,indexes,Γ,compressionIndexes,DC,IDC)

Equilibrium error function: returns deviations from equilibrium around steady state.

Split computation into *Aggregate Part*, handled by [`Fsys_agg()`](@ref),
and *Heterogeneous Agent Part*.

# Arguments
- `X`,`XPrime`: deviations from steady state in periods t [`X`] and t+1 [`XPrime`]
- `Xss`: states and controls in steady state
- `Γ`,`DC`,`IDC`: transformation matrices to retrieve marginal distributions [`Γ`] and
    marginal value functions [`DC`,`IDC`] from deviations
- `indexes`,`compressionIndexes`: access `Xss` by variable names
    (DCT coefficients of compressed ``V_m`` and ``V_k`` in case of `compressionIndexes`)

# Example
```jldoctest
julia> # Solve for steady state, construct Γ,DC,IDC as in SGU()
julia> Fsys(zeros(ntotal),zeros(ntotal),XSS,m_par,n_par,indexes,Γ,compressionIndexes,DC,IDC)
*ntotal*-element Array{Float64,1}:
 0.0
 0.0
 ...
 0.0
```
"""
function Fsys(X::AbstractArray, XPrime::AbstractArray, Xss::Array{Float64,1}, m_par,
              n_par::NumericalParameters, indexes, Γ::Array{Array{Float64,2},1},
              compressionIndexes::Array{Int,1}, DC1::Array{Float64,2},
              DC2::Array{Float64,2}, Copula::Function;Fsys_agg::Function = Fsys_agg,ret_pol_fcts = false, balanced_budget = false)
              # The function call with Duals takes
              # Reserve space for error terms
    F = zeros(eltype(X),size(X))

    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################
    # rougly 10% of computing time, more if uncompress is actually called

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(Xss[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(Xss[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

    @generate_equations(aggr_names_ess)

    ############################################################################
    # I.2. Distributions (Γ-multiplying makes sure that they are distributions)
    ############################################################################
    distr_k       = Xss[indexes.distr_k_SS] .+ Γ[1] * X[indexes.distr_k]
    distr_k_Prime = Xss[indexes.distr_k_SS] .+ Γ[1] * XPrime[indexes.distr_k]
    distr_y       = Xss[indexes.distr_y_SS] .+ Γ[2] * X[indexes.distr_y]
    distr_y_Prime = Xss[indexes.distr_y_SS] .+ Γ[2] * XPrime[indexes.distr_y]

    # Joint distributions (uncompressing)
    CDF_k         = cumsum([0.0; distr_k[:]])
    CDF_y         = cumsum([0.0; distr_y[:]])
    CDF_joint     = Copula(CDF_k[:], CDF_y[:]) # roughly 5% of time

    ##### requires Julia 1.1
    distr         = diff(diff(CDF_joint; dims=2);dims=1)
    ############################################################################
    # I.3 uncompressing policies/value functions
    ###########################################################################
     if any((tot_dual.(XPrime[indexes.Vk])+realpart.(XPrime[indexes.Vk])).!= 0.0)
        θk      = uncompress(compressionIndexes, XPrime[indexes.Vk], DC1,DC2, n_par)
        VkPrime = Xss[indexes.VkSS]+  θk
     else
         VkPrime = Xss[indexes.VkSS].+ zeros(eltype(X),1)[1]
     end
    VkPrime .= (exp.(VkPrime))

    ############################################################################
    #           II. Auxiliary Variables                                        #
    ############################################################################
    # Transition Matrix Productivity
    if tot_dual.(σ .+ zeros(eltype(X),1)[1])==0.0
        if σ==1.0
            Π                  = n_par.Π .+ zeros(eltype(X),1)[1]
        else
            Π                  = n_par.Π
            PP                 =  ExTransition(m_par.ρ_h,n_par.bounds_y,sqrt(σ))
            Π[1:end-1,1:end-1] = PP.*(1.0-m_par.ζ)
        end
    else
        Π                  = n_par.Π .+ zeros(eltype(X),1)[1]
        PP                 =  ExTransition(m_par.ρ_h,n_par.bounds_y,sqrt(σ))
        Π[1:end-1,1:end-1] = PP.*(1.0-m_par.ζ)
    end

    ############################################################################
    #           III. Error term calculations (i.e. model starts here)          #
    ############################################################################

    ############################################################################
    #           III. 1. Aggregate Part #
    ############################################################################
    F            = Fsys_agg(X, XPrime, Xss, distr, m_par, n_par, indexes)

    # Error Term on prices/aggregate summary vars (logarithmic, controls)
    KP           = dot(n_par.grid_k,distr_k[:])
    F[indexes.K] = log.(K)     - log.(KP)
    F[indexes.B] = log.(RLPrime./πPrime) .- log.((rPrime .- 1.0 .+ qPrime)./q)

    # Average Human Capital =
    # average productivity (at the productivit grid, used to normalize to 0)
    H       = H_fnc(n_par.grid_y,distr_y,m_par)

    ############################################################################
    #               III. 2. Heterogeneous Agent Part                           #
    ############################################################################
    # Incomes
    incgross, inc,~,~,tax_prog_scale,meshes = OneAssetincomes(n_par,m_par,distr,N,r,B/K,w,profits,A,RL,π,mcw,q,τprog,τlev,Ht,H)
    if balanced_budget
        # Rebate government spending lump-sum to all households
        inc[1] .= inc[1] .+ G
    end
    # Calculate optimal policies
    # expected margginal values
    EVkPrime = reshape(VkPrime,(n_par.nk, n_par.ny))
    EVkPrime = EVkPrime*Π'

    # roughly 20% time
    c_star, k_star = EGM_OneAssetpolicyupdate(EVkPrime ,q,π,RL .*A,B/K,BPrime/KPrime,r .- 1.0,inc[1],n_par,m_par, meshes) # policy iteration
    if(ret_pol_fcts)
        return c_star, k_star, inc, incgross, distr
    end
    # Update marginal values
    asset_ret = ((r .- 1.0) .+ q .+ B.*RL ./ (K .* π))./(qlag .+ B ./ K)
    Vk_new = asset_ret .* mutil(c_star,m_par.ξ)

    # roughly 20% time
    # Calculate error terms on marginal values
    Vk_err        =log.((Vk_new)) .- reshape(Xss[indexes.VkSS],(n_par.nk,n_par.ny))
    Vk_thet       = compress(compressionIndexes, Vk_err, DC1,DC2)
    F[indexes.Vk] = X[indexes.Vk] .- Vk_thet

    # roughly 20% time
    # Error Term on distribution (in levels, states)
    dPrime        = DirectTransition(k_star,Π,n_par,distr)
    dPrs          = reshape(dPrime,n_par.nk,n_par.ny)
    temp          = dropdims(sum(dPrs,dims=2),dims=2)
    F[indexes.distr_k] = temp[1:end-1] - distr_k_Prime[1:end-1]
    temp          = distr_y'*Π# dropdims(sum(dPrs,dims=(1,2)),dims=(1,2))
    F[indexes.distr_y] = temp[1:end-1] - distr_y_Prime[1:end-1]

    # ToDo: distr_summary for one asset

    Htact       = dot(distr_y[1:end],([n_par.grid_y[1:end-1];m_par.y_e]/H).^(tax_prog_scale))
    F[indexes.Ht]     =log.(Ht) - log.(Htact)

    F[indexes.GiniX]    = log.(GiniX)   - Xss[indexes.GiniXSS]
    F[indexes.I90share]   = log.(I90share)  - Xss[indexes.I90shareactSS]
    F[indexes.I90sharenet]   = log.(I90sharenet)  - Xss[indexes.I90sharenetactSS]

    F[indexes.w90share] = log.(w90share)  - Xss[indexes.w90shareactSS]
    F[indexes.GiniW]    = log.(GiniW)   - Xss[indexes.GiniWactSS]
    F[indexes.GiniC]    = log.(GiniC)   - Xss[indexes.GiniCactSS]
    F[indexes.sdlgC]    = log.(sdlgC)   - Xss[indexes.sdlgCactSS]
    F[indexes.P9010C]   = log.(P9010C)  - Xss[indexes.P9010CactSS]
    F[indexes.P9010I]   = log.(P9010I)  - Xss[indexes.P9010IactSS]
    F[indexes.GiniI]    = log.(GiniI)   - Xss[indexes.GiniIactSS]
    F[indexes.P90C]   = log.(P90C)  - Xss[indexes.P90CactSS]
    F[indexes.P50C]   = log.(P50C)  - Xss[indexes.P50CactSS]
    F[indexes.P10C]   = log.(P10C)  - Xss[indexes.P10CactSS]

    return F
end
