@doc raw"""
    find_OneAssetSS(refined_ny)

Find the stationary equilibrium using a final resolution of `refined_ny` for income.

# Returns
- `XSS::Array{Float64,1}`, `XSSaggr::Array{Float64,1}`: steady state vectors produced by [`@writeXSS()`](@ref)
- `indexes`, `indexes_aggr`: `struct`s for accessing `XSS`,`XSSaggr` by variable names, produced by [`@make_fn()`](@ref),
        [`@make_fnaggr()`](@ref)
- `compressionIndexes::Array{Array{Int,1},1}`: indexes for compressed marginal value functions (``V_m`` and ``V_k``)
- `Copula(x,y,z)`: function that maps marginals `x`,`y`,`z` to approximated joint distribution, produced by
        [`mylinearinterpolate3()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`
- `distrSS::Array{Float64,3}`: steady state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
- `CDF_SS`, `CDF_m`, `CDF_k`, `CDF_y`: cumulative distribution functions (joint and marginals)
"""
function find_OneAssetSS(state_names,control_names,BtoK;ModelParamStruct = ModelParameters,flattenable = flattenable, path)

        BLAS.set_num_threads(Threads.nthreads())

        # global m_par, n_par, CDF_m, CDF_k, CDF_y
        # load estimated parameter set
        m_par           = ModelParamStruct(λ = 1.0)
        @load string(path,"/",e_set.mode_start_file) par_final parnames
        par = par_final[1:length(parnames)]
        if e_set.me_treatment != :fixed
        m_par = Flatten.reconstruct(m_par, par[1:length(par) - length(e_set.meas_error_input)],flattenable)
        else
        m_par = Flatten.reconstruct(m_par, par, flattenable)
        end

        # Read out numerical parameters for starting guess solution with reduced income grid.
        ny              = 5; # eigs in Ksupply quickly increases in runtime in ny (more than ny^2).
        grid_y, Π, bounds = Tauchen(m_par.ρ_h,ny) # Income grid and transitions
        # Include entrepreneurs into the income transitions
        Π               = [Π .* (1.0 .- m_par.ζ)  m_par.ζ .* ones(ny);
                        m_par.ι ./ ny * ones(1,ny) 1.0 .- m_par.ι]
        grid_y          = [exp.(grid_y .* m_par.σ_h ./ sqrt(1.0 .- m_par.ρ_h.^2));
                        (m_par.ζ .+ m_par.ι)/m_par.ζ]
        # Calculate expected level of human capital
        Paux            = Π^1000
        H               = H_fnc(grid_y,Paux[1,:],m_par)
        # Numerical parameters
        n_par           = NumericalParameters(ny = ny+1, bounds_y = bounds, grid_y = grid_y, Π = Π, H = H)

        # -------------------------------------------------------------------------------
        ## STEP 1: Find the stationary equilibrium
        # -------------------------------------------------------------------------------
        # Capital stock guesses
        Kmax      = 5.0 * ((m_par.δ_0 - 0.0025 + (1.0 - m_par.β) / m_par.β) / m_par.α)^(1.0 / (m_par.α - 1.0))
        Kmin      = 1.0 * ((m_par.δ_0 - 0.0005 + (1.0 - m_par.β) / m_par.β) / m_par.α)^(0.5 / (m_par.α - 1.0))
        K         = range(Kmin, stop = Kmax, length = 8)
        # a.) Define excess demand function
        d(K)      = OneAssetKdiff(K,BtoK,n_par,m_par)

        # b.) Find equilibrium capital stock (multigrid on y)
        # ba.) initial calculation
        KSS       = Brent(d, .9*Kmin, 1.2*Kmax)[1]

        # c.) Calculate other equilibrium quantities
        NSS       = employment(KSS, 1.0 ./ (m_par.μ*m_par.μw), m_par)
        rSS       = interest(KSS,1.0 / m_par.μ, NSS, m_par)
        wSS       = wage(KSS,1.0 / m_par.μ, NSS , m_par)
        YSS       = output(KSS,1.0,NSS, m_par)
        ProfitsSS = profitsSS_fnc(YSS,m_par.RB,m_par)
        println("first ProfitSS: ",ProfitsSS)

        KSS, TransitionMatSS, distrSS,
                c_starSS, k_starSS,VkSS =
                OneAssetKsupply(m_par.RB,1.0+ rSS,wSS*NSS/n_par.H,ProfitsSS,BtoK,n_par,m_par)
        println("first KSS: ", KSS)
        println("first KSS/YSS: ",KSS/YSS)
        ## bb.) refinement
        ny              = n_par.ny_refined; # eigs in Ksupply quickly increases in runtime in ny
        grid_y, Π, bounds= Tauchen(m_par.ρ_h,ny)
        gy              = [exp.(grid_y .* m_par.σ_h ./ sqrt(1.0 .- m_par.ρ_h.^2)); (m_par.ζ .+
                        m_par.ι)/m_par.ζ]
        # Include entrepreneurs into the income transitions
        Π               = [Π .* (1.0 .- m_par.ζ)  m_par.ζ .* ones(ny);
                        m_par.ι ./ ny * ones(1,ny) 1.0 .- m_par.ι]
        grid_y          = [exp.(grid_y .* m_par.σ_h ./ sqrt(1.0 .- m_par.ρ_h.^2));
                        (m_par.ζ .+ m_par.ι)/m_par.ζ]
        Paux            = Π^1000
        H               = H_fnc(grid_y,Paux[1,:],m_par)

        # Write changed parameter values to n_par
        @set! n_par.ny          = ny + 1
        @set! n_par.nstates     = (ny + 1) + n_par.nk + length(state_names) - 2
        @set! n_par.naggrstates = length(state_names)
        @set! n_par.naggrcontrols = length(control_names)
        @set! n_par.aggr_names  = [state_names; control_names]
        @set! n_par.naggr       = length(n_par.aggr_names)

        @set! n_par.bounds_y    = bounds
        @set! n_par.ϵ           = 1e-10
        @set! n_par.grid_y      = grid_y
        @set! n_par.Π           = Π
        @set! n_par.H           = H

        KSS                     = Brent(d, KSS*.9, KSS*1.2)[1]

        # c.) Calculate other equilibrium quantities
        NSS                     = employment(KSS, 1.0 ./ (m_par.μ*m_par.μw), m_par)
        rSS                     = interest(KSS,1.0 / m_par.μ, NSS, m_par)
        wSS                     = wage(KSS,1.0 / m_par.μ, NSS , m_par)
        YSS                     = output(KSS,1.0,NSS, m_par)
        ProfitsSS               = profitsSS_fnc(YSS,m_par.RB,m_par)
        RLSS                    = m_par.RB
        println("2nd ProfitSS: ",ProfitsSS)
        KSS, TransitionMatSS, distrSS,
                c_starSS, k_starSS, VkSS =
                OneAssetKsupply(RLSS,1.0+ rSS,wSS*NSS/n_par.H,ProfitsSS,BtoK,n_par,m_par)
        println("2nd KSS: ",KSS)
        println("KSS/YSS: ",KSS/YSS)
        VkSS                    = log.(VkSS)
        RBSS                    = m_par.RB
        ISS                     = m_par.δ_0*KSS

        # Produce distributional summary statistics
        BSS = BtoK * KSS
        qΠSS            =qΠSS_fnc(YSS,RBSS,m_par)
        liquidvalue = value_liquid(BSS,qΠSS,qΠSS,m_par)
        incgross, inc, av_tax_rateSS, taxrev = OneAssetincomes(n_par,m_par,distrSS,NSS,1 .+ rSS,BtoK,liquidvalue,wSS,ProfitsSS,1.0,RLSS,m_par.π,1.0 ./ m_par.μw,1.0,m_par.τ_prog,m_par.τ_lev,1.0,H)
        println("av_tax_rateSS: ",av_tax_rateSS)
        TSS           = (distrSS[:]' * taxrev[:] + av_tax_rateSS*((1.0 .- 1.0 ./ m_par.μw).*wSS.*NSS))
        println("TSS: ",TSS)
        println("BSS: ",BSS)
        println("qΠSS: ",qΠSS_fnc(YSS,m_par.RB,m_par) .- 1.0)
        BgovSS        = BSS .- qΠSS_fnc(YSS,m_par.RB,m_par) .+ 1.0
        println("BgovSS: ", BgovSS)
        GSS           = TSS - (m_par.RB./m_par.π-1.0)*BgovSS
        
        distr_k_SS, distr_y_SS, GiniWSS, I90shareSS,I90sharenetSS, GiniXSS,
                sdlogxSS, P9010CSS, GiniCSS, sdlgCSS, P9010ISS, GiniISS, sdlgISS, w90shareSS, P10CSS, P50CSS, P90CSS =
                distrSummaries(distrSS, c_starSS, n_par, inc,incgross,1.0)
        # ------------------------------------------------------------------------------
        ## STEP 2: Dimensionality reduction
        # ------------------------------------------------------------------------------
        # 2a.) Discrete cosine transformation of policies or MUs
        ThetaVk   = dct(VkSS)[:] # Discrete Cosine transformation
        ind             = sortperm(abs.(ThetaVk[:]);rev=true) #Indexes of sorted coefficients
        coeffs          = 1;
        # Find the important basis functions (discrete cosine) for c_polSS
        while norm(ThetaVk[ind[1:coeffs]])/norm(ThetaVk ) < 1 - n_par.reduc
                coeffs += 1
        end
        compressionIndexes  = ind[1:coeffs]
        # 2b.) Produce the Copula as an interpolant on the distribution function
        #      and its marginals
        CDF_SS     = zeros(n_par.nk+1,n_par.ny+1)
        CDF_SS[2:end,2:end]     = cumsum(cumsum(distrSS,dims=1),dims=2)
        distr_k_SS = sum(distrSS,dims=2)[:]
        distr_y_SS = sum(distrSS,dims=1)[:]
        CDF_k      = cumsum([0.0; distr_k_SS[:]])
        CDF_y      = cumsum([0.0; distr_y_SS[:]])

        Copula(x::Vector,y::Vector) = mylinearinterpolate2(CDF_k, CDF_y,
                                                                CDF_SS, x, y)

        # ------------------------------------------------------------------------------

        @include "../input_aggregate_steady_state.jl"

        # write to XSS vector
        @writeOneAssetXSS state_names control_names
        # produce indexes to access XSS etc.
        indexes = produce_OneAssetindexes(n_par, compressionIndexes)
        indexes_aggr = produce_indexes_aggr(n_par)
        ntotal                  = indexes.profits
        @set! n_par.ntotal      = ntotal
        @set! n_par.ncontrols   = length(compressionIndexes) + n_par.naggrcontrols

        return XSS, XSSaggr, indexes, indexes_aggr, compressionIndexes, Copula, n_par, #=
                =# m_par, CDF_SS, CDF_k, CDF_y, distrSS
end
