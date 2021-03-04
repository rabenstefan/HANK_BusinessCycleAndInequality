
function VFI(Kgrid,BtoK,m_par)
        # construct (composite) consumption over states (Kgrid)
        N = employment.(Kgrid,1.0 / (m_par.μ * m_par.μw),m_par)
        w = wage.(Kgrid,1.0/m_par.μ,N,m_par)
        Y = output.(Kgrid,1.0,N,m_par)
        qΠ = qΠSS_fnc.(Y,m_par)
        Rtot = Rtot_fnc.(Kgrid,BtoK,qΠ,Y,N,m_par)
        Π = Y .* (1.0 .- 1/m_par.μ)
        y = (1.0 .- m_par.τ_lev).*(w.*N).^(1.0 .- m_par.τ_prog) * (m_par.γ + m_par.τ_prog)/(m_par.γ + 1.0)
        x = (y .+ (Rtot .- m_par.δ_0).*Kgrid .+ Π) .- Kgrid'
        u = x.^ (1.0 .- m_par.ξ)
        # value function iteration
        V = ones(size(Kgrid))
        lastV = V.+1
        while norm(V-lastV,Inf)>1.0e-16
                lastV = V
                V = maximum(u .+ m_par.β*lastV',dims=2)
        end
        # find capital level that maps to itself
        Kopt = [Kgrid[cartix[2]] for cartix in argmax(u .+ m_par.β*V',dims=2)]
        zeroix = findall(isapprox.(Kopt.-Kgrid,0.0;atol=1.0e-8))
        return Kgrid[zeroix]
end

@doc raw"""
    find_RASS(state_names,control_names;ModelParamStruct,flattenable,path)

Find the stationary equilibrium of the representative agent model.

# Returns
- `XSS::Array{Float64,1}`, `XSSaggr::Array{Float64,1}`: steady state vectors produced by [`@writeXSS()`](@ref)
- `indexes`, `indexes_aggr`: `struct`s for accessing `XSS`,`XSSaggr` by variable names, produced by [`@make_fn()`](@ref),
        [`@make_fnaggr()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`
"""
function find_SS(state_names,control_names,BtoK;ModelParamStruct = ModelParameters,flattenable = flattenable, path)

        BLAS.set_num_threads(Threads.nthreads())

        # global m_par, n_par, CDF_m, CDF_k, CDF_y
        n_par           = NumericalParameters()
        @set! n_par.nstates     = (ny + 1) + n_par.nk + n_par.nm + length(state_names) - 3
        @set! n_par.naggrstates = length(state_names)
        @set! n_par.naggrcontrols = length(control_names)
        @set! n_par.aggr_names  = [state_names; control_names]
        @set! n_par.naggr       = length(n_par.aggr_names)
        # load estimated parameter set
        m_par           = ModelParamStruct( )
        @load string(path,"/",e_set.mode_start_file) par_final parnames
        par = par_final[1:length(parnames)]
        if e_set.me_treatment != :fixed
        m_par = Flatten.reconstruct(m_par, par[1:length(par) - length(e_set.meas_error_input)],flattenable)
        else
        m_par = Flatten.reconstruct(m_par, par, flattenable)
        end
        # -------------------------------------------------------------------------------
        ## STEP 1: Find the stationary equilibrium
        # -------------------------------------------------------------------------------
        # Capital stock guesses
        Kmax      = 2.0 * ((m_par.δ_0 - 0.0025 + (1.0 - m_par.β) / m_par.β) / m_par.α)^(1.0 / (m_par.α - 1.0))
        Kmin      = 1.0 * ((m_par.δ_0 - 0.0005 + (1.0 - m_par.β) / m_par.β) / m_par.α)^(0.5 / (m_par.α - 1.0))
        K         = range(.9*Kmin, stop = 1.2*Kmax, length = 1000)
        KSS = VFI(K,BtoK,m_par)
        # c.) Calculate other equilibrium quantities
        NSS                     = employment(KSS, 1.0 ./ (m_par.μ*m_par.μw), m_par)
        rSS                     = interest(KSS,1.0 / m_par.μ, NSS, m_par)
        wSS                     = wage(KSS,1.0 / m_par.μ, NSS , m_par)
        YSS                     = output(KSS,1.0,NSS, m_par)
        ProfitsSS               = profitsSS_fnc(YSS,m_par)
        BSS = BtoK*KSS
        RLSS                    = RLSS_fnc(YSS,BSS,m_par)
        RBSS                    = m_par.RB
        ISS                     = m_par.δ_0*KSS
        av_tax_rateSS, taxrev
        BgovSS        = BSS .- qΠSS_fnc(YSS,m_par)
        GSS           = TSS - (m_par.RB./m_par.π-1.0)*BgovSS
end

        # a.) Define excess demand function
        d(K)      = Kdiff(K,n_par,m_par)

        # b.) Find equilibrium capital stock (multigrid on y)
        # ba.) initial calculation
        KSS       = Brent(d, .9*Kmin, 1.2*Kmax)[1]

        # c.) Calculate other equilibrium quantities
        NSS       = employment(KSS, 1.0 ./ (m_par.μ*m_par.μw), m_par)
        rSS       = interest(KSS,1.0 / m_par.μ, NSS, m_par)
        wSS       = wage(KSS,1.0 / m_par.μ, NSS , m_par)
        YSS       = output(KSS,1.0,NSS, m_par)
        ProfitsSS = profitsSS_fnc(YSS,m_par)
        println("first ProfitSS: ",ProfitsSS)

        KSS, BSS, TransitionMatSS,TransitionMatSS_a,TransitionMatSS_n, distrSS,
                c_a_starSS, m_a_starSS, k_a_starSS, c_n_starSS, m_n_starSS,VmSS, VkSS =
                Ksupply(RLSS_fnc(YSS,sum(n_par.dist_guess[:] .* n_par.mesh_m[:]),m_par),1.0+ rSS,wSS*NSS/n_par.H,ProfitsSS,n_par,m_par)
        println("first BSS: ", BSS)
        println("first BSS/YSS: ",BSS/YSS)
        ## bb.) refinement
        ny              = n_par.ny_refined; # eigs in Ksupply quickly increases in runtime in ny
        grid_y, Π, bounds= Tauchen(m_par.ρ_h,ny)
        gy              = [exp.(grid_y .* m_par.σ_h ./ sqrt(1.0 .- m_par.ρ_h.^2)); (m_par.ζ .+
                        m_par.ι)/m_par.ζ]
        # Interpolate distribution on refined wealth-income grid
        refined_dist    = mylinearinterpolate3(n_par.grid_m,n_par.grid_k, n_par.grid_y,
                                                distrSS,n_par.grid_m,n_par.grid_k, gy)
        refined_dist    = refined_dist ./ sum(refined_dist,dims=(1,2,3))
        # Include entrepreneurs into the income transitions
        Π               = [Π .* (1.0 .- m_par.ζ)  m_par.ζ .* ones(ny);
                        m_par.ι ./ ny * ones(1,ny) 1.0 .- m_par.ι]
        grid_y          = [exp.(grid_y .* m_par.σ_h ./ sqrt(1.0 .- m_par.ρ_h.^2));
                        (m_par.ζ .+ m_par.ι)/m_par.ζ]
        Paux            = Π^1000
        H               = H_fnc(grid_y,Paux[1,:],m_par)

        # Write changed parameter values to n_par
        @set! n_par.ny          = ny + 1
        @set! n_par.nstates     = (ny + 1) + n_par.nk + n_par.nm + length(state_names) - 3
        @set! n_par.naggrstates = length(state_names)
        @set! n_par.naggrcontrols = length(control_names)
        @set! n_par.aggr_names  = [state_names; control_names]
        @set! n_par.naggr       = length(n_par.aggr_names)

        @set! n_par.bounds_y    = bounds
        @set! n_par.ϵ           = 1e-10
        @set! n_par.grid_y      = grid_y
        @set! n_par.mesh_y      = repeat(reshape(grid_y, (1, 1, ny + 1)), outer=[n_par.nm, n_par.nk, 1])
        @set! n_par.mesh_m      = repeat(reshape(n_par.grid_m, (n_par.nm, 1, 1)),
                                outer=[1, n_par.nk, ny + 1])
        @set! n_par.mesh_k      = repeat(reshape(n_par.grid_k, (1, n_par.nk, 1)),
                                        outer=[n_par.nm, 1, ny + 1])
        @set! n_par.Π           = Π
        @set! n_par.H           = H
        @set! n_par.dist_guess  = refined_dist

        KSS                     = Brent(d, KSS*.9, KSS*1.2)[1]

        # c.) Calculate other equilibrium quantities
        NSS                     = employment(KSS, 1.0 ./ (m_par.μ*m_par.μw), m_par)
        rSS                     = interest(KSS,1.0 / m_par.μ, NSS, m_par)
        wSS                     = wage(KSS,1.0 / m_par.μ, NSS , m_par)
        YSS                     = output(KSS,1.0,NSS, m_par)
        ProfitsSS               = profitsSS_fnc(YSS,m_par)
        RLSS                    = RLSS_fnc(YSS,BSS,m_par)
        println("2nd ProfitSS: ",ProfitsSS)
        KSS, BSS, TransitionMatSS, TransitionMatSS_a, TransitionMatSS_n, distrSS,
                c_a_starSS, m_a_starSS, k_a_starSS, c_n_starSS, m_n_starSS,VmSS, VkSS =
                Ksupply(RLSS,1.0+ rSS,wSS*NSS/n_par.H,ProfitsSS,n_par,m_par)
        println("2nd BSS: ",BSS)
        println("BSS/YSS: ",BSS/YSS)
        VmSS                    = log.(VmSS)
        VkSS                    = log.(VkSS)
        RBSS                    = m_par.RB
        ISS                     = m_par.δ_0*KSS

        # Produce distributional summary statistics
        incgross, inc, av_tax_rateSS, taxrev = incomes(n_par,m_par,distrSS,NSS,1 .+ rSS,wSS,ProfitsSS,1.0,RLSS,m_par.π,1.0 ./ m_par.μw,1.0,m_par.τ_prog,m_par.τ_lev,1.0,H)
        println("av_tax_rateSS: ",av_tax_rateSS)
        TSS           = (distrSS[:]' * taxrev[:] + av_tax_rateSS*((1.0 .- 1.0 ./ m_par.μw).*wSS.*NSS))
        println("TSS: ",TSS)
        println("qΠSS: ",qΠSS_fnc(YSS,m_par))
        BgovSS        = BSS .- qΠSS_fnc(YSS,m_par)
        println("BgovSS: ", BgovSS)
        GSS           = TSS - (m_par.RB./m_par.π-1.0)*BgovSS

        distr_m_SS, distr_k_SS, distr_y_SS, share_borrowerSS, GiniWSS, I90shareSS,I90sharenetSS, GiniXSS,
                sdlogxSS, P9010CSS, GiniCSS, sdlgCSS, P9010ISS, GiniISS, sdlgISS, w90shareSS, P10CSS, P50CSS, P90CSS =
                distrSummaries(distrSS, c_a_starSS, c_n_starSS, n_par, inc,incgross, m_par)
        # ------------------------------------------------------------------------------
        ## STEP 2: Dimensionality reduction
        # ------------------------------------------------------------------------------
        # 2a.) Discrete cosine transformation of policies or MUs
        aux             = dct(VmSS)
        ThetaVm   = aux[:] # Discrete Cosine transformation
        ind             = sortperm(abs.(ThetaVm[:]);rev=true) #Indexes of sorted coefficients
        coeffs          = 1
        # Find the important basis functions (discrete cosine) for c_polSS
        while norm(ThetaVm[ind[1:coeffs]])/norm(ThetaVm ) < 1 - n_par.reduc
                coeffs += 1
        end
        compressionIndexesVm  = ind[1:coeffs]

        ThetaVk   = dct(VkSS)[:] # Discrete Cosine transformation
        ind             = sortperm(abs.(ThetaVk[:]);rev=true) #Indexes of sorted coefficients
        coeffs          = 1;
        # Find the important basis functions (discrete cosine) for c_polSS
        while norm(ThetaVk[ind[1:coeffs]])/norm(ThetaVk ) < 1 - n_par.reduc
                coeffs += 1
        end
        compressionIndexesVk  = ind[1:coeffs]

        compressionIndexes =Array{Array{Int,1},1}(undef ,2)
        compressionIndexes[1] = compressionIndexesVm
        compressionIndexes[2] = compressionIndexesVk
        # 2b.) Produce the Copula as an interpolant on the distribution function
        #      and its marginals
        CDF_SS     = zeros(n_par.nm+1,n_par.nk+1,n_par.ny+1)
        CDF_SS[2:end,2:end,2:end]     = cumsum(cumsum(cumsum(distrSS,dims=1),dims=2),dims=3)
        distr_m_SS = sum(distrSS,dims=(2,3))[:]
        distr_k_SS = sum(distrSS,dims=(1,3))[:]
        distr_y_SS = sum(distrSS,dims=(1,2))[:]
        CDF_m      = cumsum([0.0; distr_m_SS[:]])
        CDF_k      = cumsum([0.0; distr_k_SS[:]])
        CDF_y      = cumsum([0.0; distr_y_SS[:]])

        Copula(x::Vector,y::Vector,z::Vector) = mylinearinterpolate3(CDF_m, CDF_k, CDF_y,
                                                                CDF_SS, x, y, z)

        # ------------------------------------------------------------------------------

        @include "../input_aggregate_steady_state.jl"

        # write to XSS vector
        @writeXSS state_names control_names
        # produce indexes to access XSS etc.
        indexes = produce_indexes(n_par, compressionIndexesVm, compressionIndexesVk)
        indexes_aggr = produce_indexes_aggr(n_par)

        return XSS, XSSaggr, indexes, indexes_aggr, compressionIndexes, Copula, n_par, #=
                =# m_par, d, CDF_SS, CDF_m, CDF_k, CDF_y, distrSS
end
