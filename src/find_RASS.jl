
function VFI(Kgrid,av_tax_rate,m_par;tol = 1.0e-16)
        # construct (composite) consumption over states (Kgrid)
        N = [employment(K,1.0 / (m_par.μ * m_par.μw),m_par) for K in Kgrid]
        w = [wage(Kgrid[i],1.0/m_par.μ,N[i],m_par) for i in 1:length(Kgrid)]
        Y = [output(Kgrid[i],1.0,N[i],m_par) for i in 1:length(Kgrid)]
        R = [interest(Kgrid[i],1.0 / m_par.μ, N[i], m_par) + 1.0 for i in 1:length(Kgrid)]
        Π = [profitsSS_fnc(Y[i],R[i],m_par) for i in 1:length(Kgrid)]
        y = (1.0 .- av_tax_rate).*(w.*N) * (m_par.γ + m_par.τ_prog)/(m_par.γ + 1.0)
        x = (y .+ R .* Kgrid .+ (1.0 .- av_tax_rate).*Π) .- Kgrid'
        u = x.^ (1.0 .- m_par.ξ)
        # value function iteration
        V = ones(size(Kgrid))
        lastV = V.+1
        while norm(V-lastV,Inf)>tol
                lastV = V
                V = maximum(u .+ m_par.β*lastV',dims=2)
        end
        # find capital level that maps to itself
        Kopt = [Kgrid[cartix[2]] for cartix in argmax(u .+ m_par.β*V',dims=2)]
        return Kgrid[argmin(abs.(Kopt-Kgrid))]
end

@doc raw"""
    find_RASS(state_names,control_names,BtoK,av_tax_rate;ModelParamStruct,flattenable,path)

Find the stationary equilibrium of the representative agent model.

# Returns
- `XSSaggr::Array{Float64,1}`: steady state vectors produced by [`@writeXSSaggr()`](@ref)
- `indexes_aggr`: `struct` for accessing `XSSaggr` by variable names, produced by [`@make_fnaggr()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`
"""
function find_RASS(state_names,control_names,BtoK,av_tax_rate;ModelParamStruct = ModelParameters,flattenable = flattenable, path)
        BLAS.set_num_threads(Threads.nthreads())

        # global m_par, n_par, CDF_m, CDF_k, CDF_y
        n_par           = NumericalParameters()
        @set! n_par.nstates     = length(state_names)
        @set! n_par.naggrstates = length(state_names)
        @set! n_par.ncontrols = length(control_names)
        @set! n_par.naggrcontrols = length(control_names)
        @set! n_par.aggr_names  = [state_names; control_names]
        @set! n_par.naggr       = length(n_par.aggr_names)
        @set! n_par.ntotal      = n_par.naggr
        # load estimated parameter set
        m_par           = ModelParamStruct( )
        @load string(path,"/",e_set.mode_start_file) par_final parnames
        par = par_final[1:length(parnames)]
        if e_set.me_treatment != :fixed
        m_par = Flatten.reconstruct(m_par, par[1:length(par) - length(e_set.meas_error_input)],flattenable)
        else
        m_par = Flatten.reconstruct(m_par, par, flattenable)
        end
        @set! m_par.τ_prog = 0.0
        # -------------------------------------------------------------------------------
        ## STEP 1: Find the stationary equilibrium
        # -------------------------------------------------------------------------------
        # Capital stock guesses
        Kmax      = 2.0 * ((m_par.δ_0 - 0.0025 + (1.0 - m_par.β) / m_par.β) / m_par.α)^(1.0 / (m_par.α - 1.0))
        Kmin      = 1.0 * ((m_par.δ_0 - 0.0005 + (1.0 - m_par.β) / m_par.β) / m_par.α)^(0.5 / (m_par.α - 1.0))
        K         = range(.9*Kmin, stop = 1.2*Kmax, length = 1000)
        KSS = VFI(collect(K),av_tax_rate,m_par)
        # c.) Calculate other equilibrium quantities
        NSS                     = employment(KSS, 1.0 ./ (m_par.μ*m_par.μw), m_par)
        rSS                     = interest(KSS,1.0 / m_par.μ, NSS, m_par)
        RBSS                    = (rSS + 1.0) * m_par.π
        wSS                     = wage(KSS,1.0 / m_par.μ, NSS , m_par)
        YSS                     = output(KSS,1.0,NSS, m_par)
        ProfitsSS               = profitsSS_fnc(YSS,RBSS,m_par)
        BSS                     = BtoK*KSS
        RLSS                    = RBSS
        ISS                     = m_par.δ_0*KSS
        av_tax_rateSS           = av_tax_rate
        BgovSS                  = BSS .- qΠSS_fnc(YSS,RBSS,m_par)
        TSS                     = av_tax_rate*(wSS*NSS + profitsSS_fnc(YSS,RBSS,m_par))
        GSS                     = TSS - (RBSS/m_par.π-1.0)*BgovSS
        @include "input_aggregate_steady_state.jl"
        # Define aggregates that concern distributions with fake values
        @setDistrSSvals
        @writeXSSaggr state_names control_names
        indexes_aggr = produce_indexes_aggr(n_par)
        return XSSaggr, indexes_aggr, n_par, m_par
end