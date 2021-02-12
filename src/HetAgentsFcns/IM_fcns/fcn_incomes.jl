function incomes(n_par,m_par,K,distr,N,r,w,Y,profits,I,RB,RL,mcw,q,τprog,Ht,H)

    # NSS       = employment(KSS, 1.0 ./ (m_par.μ * m_par.μw), m_par)
    # rSS       = interest(KSS, 1.0 / m_par.μ, NSS, m_par)
    # wSS       = wage(KSS, 1.0 / m_par.μ, NSS, m_par)
    # YSS       = output(KSS, 1.0, NSS, m_par)
    # ProfitsSS = (1.0 -1.0 / m_par.μ) .* YSS
    # ISS       = m_par.δ_0 * KSS
    # RBSS      = m_par.RB
    
    
    eff_int   = (RL .+ (m_par.Rbar .* (n_par.mesh_m.<=0.0))) # effective rate
    tax_prog_scale = (m_par.γ + m_par.τ_prog)/((m_par.γ + τprog))
    incgross = Array{Array{Float64, 3}}(undef, 6)
    inc = Array{Array{Float64, 3}}(undef, 6)
    # mcw = 1.0 ./ m_par.μw
    incgross =[  mcw .*w .*N ./(Ht) .*(n_par.mesh_y ./H) .^tax_prog_scale .+
        (1.0 .- mcw).*w.*N,# labor income (NEW)
        (r .- 1.0).* n_par.mesh_k, # rental income
        eff_int .* n_par.mesh_m, # liquid asset Income
        n_par.mesh_k .* q,
        mcw .*w .*N ./(Ht) .*(n_par.mesh_y ./H) .^tax_prog_scale] # capital liquidation Income (q=1 in steady state)
    incgross[1][:,:,end].= n_par.mesh_y[:,:,end] .* profits  # profit income net of taxes
    incgross[5][:,:,end].= n_par.mesh_y[:,:,end] .* profits  # profit income net of taxes

    inc =[  (((m_par.γ - m_par.τ_prog)/(m_par.γ+1)).*m_par.τ_lev.*(n_par.mesh_y.*1.0 ./m_par.μw.*wSS.*NSS./n_par.H).^(1.0-m_par.τ_prog)).+
        ((1.0 .- 1.0 ./ m_par.μw).*wSS.*NSS),# labor income (NEW)
        interest(KSS,1.0 / m_par.μ, NSS, m_par).* n_par.mesh_k, # rental income
        eff_int .* n_par.mesh_m, # liquid asset Income
        n_par.mesh_k,
        m_par.τ_lev.*((1.0 ./ m_par.μw).*wSS.*NSS.*n_par.mesh_y./n_par.H).^(1.0-m_par.τ_prog).*((1.0 - m_par.τ_prog)/(m_par.γ+1)),
        m_par.τ_lev.*((1.0 ./ m_par.μw).*wSS.*NSS.*n_par.mesh_y./n_par.H).^(1.0-m_par.τ_prog)] # capital liquidation Income (q=1 in steady state)
    inc[1][:,:,end].= m_par.τ_lev.*(n_par.mesh_y[:,:,end] .* ProfitsSS).^(1.0-m_par.τ_prog)  # profit income net of taxes
    inc[5][:,:,end].= 0.0
    inc[6][:,:,end].= m_par.τ_lev.*(n_par.mesh_y[:,:,end] .* ProfitsSS).^(1.0-m_par.τ_prog)  # profit income net of taxes

    inc =[  GHHFA.*τlev.*((n_par.mesh_y/H).^tax_prog_scale .*mcw.*w.*N./(Ht)).^(1.0-τprog).+
            (unionprofits).*(1.0 .- av_tax_rate),# labor income (NEW)
            (r .- 1.0).* n_par.mesh_k, # rental income
            eff_int .* n_par.mesh_m, # liquid asset Income
            n_par.mesh_k .* q,
            τlev.*(mcw.*w.*N.*n_par.mesh_y./ H).^(1.0-τprog).*((1.0 - τprog)/(m_par.γ+1)),
            τlev.*((n_par.mesh_y/H).^tax_prog_scale .*mcw.*w.*N./(Ht)).^(1.0-τprog)] # capital liquidation Income (q=1 in steady state)
    inc[1][:,:,end].= τlev.*(n_par.mesh_y[:,:,end] .* profits).^(1.0-τprog) # profit income net of taxes
    inc[5][:,:,end].= 0.0
    inc[6][:,:,end].= τlev.*(n_par.mesh_y[:,:,end] .* profits).^(1.0-τprog) # profit income net of taxes

    taxrev        = incgross[5]-inc[6]
    incgrossaux   = incgross[5]
    av_tax_rateSS = (distrSS[:]' * taxrev[:])./(distrSS[:]' * incgrossaux[:])

# apply taxes to union profits
    inc[1] =(((m_par.γ - m_par.τ_prog)/(m_par.γ+1)).*m_par.τ_lev.*(n_par.mesh_y.*1.0 ./m_par.μw.*wSS.*NSS./n_par.H).^(1.0-m_par.τ_prog)).+ # labor income
        ((1.0 .- 1.0 ./ m_par.μw).*wSS.*NSS).*(1.0 .- av_tax_rateSS)# labor union income
    inc[1][:,:,end].= m_par.τ_lev.*(n_par.mesh_y[:,:,end] .* ProfitsSS).^(1.0-m_par.τ_prog)
    inc[6] =(m_par.τ_lev.*(n_par.mesh_y.*1.0 ./m_par.μw.*wSS.*NSS./n_par.H).^(1.0-m_par.τ_prog)).+ # labor income
        ((1.0 .- 1.0 ./ m_par.μw).*wSS.*NSS).*(1.0 .- av_tax_rateSS)# labor union income
    inc[6][:,:,end].= m_par.τ_lev.*(n_par.mesh_y[:,:,end] .* ProfitsSS).^(1.0-m_par.τ_prog)

    #!!!!!!!!!!!!!!!!non-steady state
    portfolio_returns = (RB .* (1.0 .- qΠlag ./ B) .+ π .* qΠlag ./ B .* (qΠ .* (1.0 .- m_par.ιΠ) .+ m_par.ωΠ .* disbursments) ./ qΠlag )
    eff_int      = (portfolio_returns  .* A .+ (m_par.Rbar .* (n_par.mesh_m.<=0.0))) ./ π  # effective rate (need to check timing below and inflation)
    eff_intPrime = (RBPrime .* APrime .+ (m_par.Rbar.*(n_par.mesh_m.<=0.0))) ./ πPrime

    GHHFA=((m_par.γ - τprog)/(m_par.γ+1)) # transformation (scaling) for composite good
    
    

    incgross =[  ((n_par.mesh_y/H).^tax_prog_scale .*mcw.*w.*N./(Ht)).+
            (unionprofits),
            (r .- 1.0).* n_par.mesh_k,                                      # rental income
            eff_int .* n_par.mesh_m,                                        # liquid asset Income
            n_par.mesh_k .* q,
            ((n_par.mesh_y/H).^tax_prog_scale .*mcw.*w.*N./(Ht))]           # capital liquidation Income (q=1 in steady state)
    incgross[1][:,:,end].= (n_par.mesh_y[:,:,end] .* profits)
    incgross[5][:,:,end].= (n_par.mesh_y[:,:,end] .* profits)

    taxrev = incgross[5]-inc[6] # tax revenues w/o tax on union profits
    incgrossaux = incgross[5]


    inc[6] = τlev.*((n_par.mesh_y/H).^tax_prog_scale .*mcw.*w.*N./(Ht)).^(1.0-τprog) .+ ((1.0 .- mcw).*w.*N).*(1.0 .- av_tax_rate)
    inc[6][:,:,end].= τlev.*(n_par.mesh_y[:,:,end] .* profits).^(1.0-τprog) # profit income net of taxes


return incgross, inc
end