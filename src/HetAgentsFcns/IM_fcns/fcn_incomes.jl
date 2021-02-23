function incomes(n_par,m_par,distr,N,r,w,profits,A,RL,π,mcw,q,τprog,τlev,Ht,H)

    # NSS       = employment(KSS, 1.0 ./ (m_par.μ * m_par.μw), m_par)
    # rSS       = interest(KSS, 1.0 / m_par.μ, NSS, m_par)
    # wSS       = wage(KSS, 1.0 / m_par.μ, NSS, m_par)
    # YSS       = output(KSS, 1.0, NSS, m_par)
    # ProfitsSS = (1.0 -1.0 / m_par.μ) .* YSS
    # ISS       = m_par.δ_0 * KSS
    # RBSS      = m_par.RB
    
    
    eff_int   = (RL .* A .+ (m_par.Rbar .* (n_par.mesh_m.<=0.0))) ./ π # effective rate
    tax_prog_scale = (m_par.γ + m_par.τ_prog)/((m_par.γ + τprog))
    GHHFA = ((m_par.γ - τprog)/(m_par.γ+1)) # transformation (scaling) for composite good
    entr_laborinc = mcw .* w .* N ./(Ht) .* (m_par.y_e ./H) .^tax_prog_scale # labor income of entrepreneur

    # mcw = 1.0 ./ m_par.μw
    incgross =[  mcw .*w .*N ./(Ht) .*(n_par.mesh_y ./H) .^tax_prog_scale .+
        (1.0 .- mcw).*w.*N,# labor income (NEW)
        (r .- 1.0).* n_par.mesh_k, # rental income
        eff_int .* n_par.mesh_m, # liquid asset Income
        n_par.mesh_k .* q,
        mcw .*w .*N ./(Ht) .*(n_par.mesh_y ./H) .^tax_prog_scale] # capital liquidation Income (q=1 in steady state)
    incgross[1][:,:,end].= n_par.mesh_y[:,:,end] .* profits .+ entr_laborinc  # profit income net of taxes
    incgross[5][:,:,end].= n_par.mesh_y[:,:,end] .* profits .+ entr_laborinc # profit income net of taxes
    
    inc = Array{Array{eltype(N), 3}}(undef, 6)
    inc[6] = τlev.*((n_par.mesh_y ./H).^tax_prog_scale .*mcw.*w.*N./(Ht)).^(1.0 .- τprog)
    inc[6][:,:,end] .= τlev.*((n_par.mesh_y[:,:,end] .* profits).^(1.0 .- τprog) .+ (entr_laborinc).^(1.0 .- τprog)) # profit income net of taxes

    taxrev = incgross[5] - inc[6] 
    av_tax_rate = (distr[:]' * taxrev[:]) ./ (distr[:]' * (incgross[5])[:])

    inc[1:5] =[  (GHHFA .* τlev.*((n_par.mesh_y ./H) .^tax_prog_scale .* mcw .*w .*N ./ Ht).^(1.0 .- τprog)).+
        ((1.0 .- mcw) .* w .*N .*(1.0 .- av_tax_rate)),# labor income (NEW)
        (r .- 1.0) .* n_par.mesh_k, # rental income
        eff_int .* n_par.mesh_m, # liquid asset Income
        n_par.mesh_k .* q, # capital liquidation Income (q=1 in steady state)
        τlev.*(mcw .*w .*N .*n_par.mesh_y ./H).^(1.0 .-τprog).*((1.0 .- τprog)/(m_par.γ+1))] 
    inc[1][:,:,end].= τlev.*((n_par.mesh_y[:,:,end] .* profits).^(1.0 .- τprog) .+ GHHFA .* (entr_laborinc).^(1.0 .- τprog))  # profit income net of taxes
    inc[5][:,:,end].= τlev.*(entr_laborinc).^(1.0 .- τprog) .* ((1.0 .- τprog)/(m_par.γ+1))

return incgross, inc, av_tax_rate, taxrev, tax_prog_scale
end