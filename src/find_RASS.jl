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
        # find KSS directly by maximizing steady state consumption
        X(K) = (1.0 - av_tax_rate) * (employment(K,1.0/(m_par.μ*m_par.μw),m_par)*wage(K,1.0/m_par.μ,employment(K,1.0/(m_par.μ*m_par.μw),m_par),m_par)*(m_par.γ + m_par.τ_prog)/(m_par.γ + 1.0) + profitsSS_fnc(output(K,1.0,employment(K,1.0/(m_par.μ*m_par.μw),m_par),m_par),interest(K,1.0/m_par.μ,employment(K,1.0/(m_par.μ*m_par.μw),m_par),m_par)+1.0,m_par)) + interest(K,1.0/m_par.μ,employment(K,1.0/(m_par.μ*m_par.μw),m_par),m_par)*K

        res = optimize(k -> -1.0*X(k),0.1,500)
        KSS = Optim.minimizer(res)
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