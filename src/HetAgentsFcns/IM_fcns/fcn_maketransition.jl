function MakeTransition(m_a_star::Array{Float64,3},
    m_n_star::Array{Float64,3},
    k_a_star::Array{Float64,3},
    Π::Array{Float64,2}, n_par::NumericalParameters)
    # create linear interpolation weights from policy functions
    idk_a, weightright_k_a, weightleft_k_a = MakeWeights(k_a_star,n_par.grid_k)
    idm_a, weightright_m_a, weightleft_m_a = MakeWeights(m_a_star,n_par.grid_m)
    idm_n, weightright_m_n, weightleft_m_n = MakeWeights(m_n_star,n_par.grid_m)

    # Adjustment case
    weight     = Array{typeof(k_a_star[1]),3}(undef, 4,n_par.ny,n_par.nk* n_par.nm*n_par.ny)
    targetindex = zeros(Int,4,n_par.ny,n_par.nk* n_par.nm*n_par.ny)
    startindex = zeros(Int,4,n_par.ny,n_par.nk* n_par.nm*n_par.ny)
    blockindex = (0:n_par.ny-1)*n_par.nk*n_par.nm
    runindex   = 0
    for zz = 1:n_par.ny # all current income states
        for kk = 1:n_par.nk # all current illiquid asset states
            for mm = 1:n_par.nm # all current liquid asset states
                runindex=runindex+1
                WLL = weightleft_m_a[mm,kk,zz] .* weightleft_k_a[mm,kk,zz]
                WRL = weightright_m_a[mm,kk,zz].* weightleft_k_a[mm,kk,zz]
                WLR = weightleft_m_a[mm,kk,zz] .* weightright_k_a[mm,kk,zz]
                WRR = weightright_m_a[mm,kk,zz].* weightright_k_a[mm,kk,zz]
                IDD = idm_a[mm,kk,zz].+(idk_a[mm,kk,zz]-1).*n_par.nm
                for jj = 1:n_par.ny
                    pp= Π[zz,jj]
                    bb =blockindex[jj]
                    weight[1,jj,runindex]     = WLL .* pp
                    weight[2,jj,runindex]     = WRL .* pp
                    weight[3,jj,runindex]     = WLR .* pp
                    weight[4,jj,runindex]     = WRR .* pp
                    targetindex[1,jj,runindex] = IDD .+ bb
                    targetindex[2,jj,runindex] = IDD + 1 .+ bb
                    targetindex[3,jj,runindex] = IDD + n_par.nm .+ bb
                    targetindex[4,jj,runindex] = IDD + n_par.nm + 1 .+ bb
                    startindex[1,jj,runindex]=runindex
                    startindex[2,jj,runindex]=runindex
                    startindex[3,jj,runindex]=runindex
                    startindex[4,jj,runindex]=runindex
                end
            end
        end
    end
    S_a         = startindex[:]
    T_a         = targetindex[:]#vcat(ti,ti+1,ti+n_par.nm,ti+n_par.nm+1)
    W_a         = weight[:]#vcat(weight1[:],weight2[:],weight3[:],weight4[:])

    # Non-Adjustment case
    weight2     = zeros(typeof(k_a_star[1]), 2,n_par.ny,n_par.nk* n_par.nm*n_par.ny)
    targetindex2 = zeros(Int, 2,n_par.ny,n_par.nk* n_par.nm*n_par.ny)
    startindex2 = zeros(Int,2,n_par.ny,n_par.nk* n_par.nm*n_par.ny)
    runindex   = 0
    for zz = 1:n_par.ny # all current income states
        for kk = 1:n_par.nk # all current illiquid asset states
            for mm = 1:n_par.nm # all current liquid asset states
                runindex=runindex+1
                WL = weightleft_m_n[mm,kk,zz]
                WR = weightright_m_n[mm,kk,zz]
                CI = idm_n[mm,kk,zz].+(kk-1).*n_par.nm
                for jj = 1:n_par.ny
                    pp = Π[zz,jj]
                    weight2[1,jj,runindex]     = WL .* pp
                    weight2[2,jj,runindex]     = WR .* pp
                    targetindex2[1,jj,runindex] = CI .+ blockindex[jj]
                    targetindex2[2,jj,runindex] = CI .+ 1 .+blockindex[jj]
                    startindex2[1,jj,runindex]=runindex
                    startindex2[2,jj,runindex]=runindex
                end
            end
        end
    end
    S_n        = startindex2[:]
    T_n        = targetindex2[:]
    W_n        = weight2[:]


    return S_a, T_a, W_a, S_n, T_n, W_n
end

function build_transition_matrix(on_grid,n_par,m_par)
    mesh_kix = repeat(1:n_par.nk,inner=n_par.nm,outer=n_par.ny)
    linix = LinearIndices((n_par.nm,n_par.nk,n_par.ny))
    liqu = [on_grid[1][1]; on_grid[1][1] .+ 1; on_grid[2][1]; on_grid[2][1] .+ 1; on_grid[3][1]; on_grid[3][1] .+ 1]
    illiqu = [mesh_kix'; mesh_kix'; on_grid[4][1]; on_grid[4][1]; on_grid[4][1] .+ 1; on_grid[4][1] .+ 1]
    # Include transition of idiosyncratic productivity.
    liqu = repeat(liqu,n_par.ny)
    illiqu = repeat(illiqu,n_par.ny)
    mesh_yix = repeat(1:n_par.ny,inner=n_par.nm*n_par.nk)
    idios = repeat(n_par.Π[mesh_yix,:]',inner=(6,1))
    Ntot = n_par.nm*n_par.nk*n_par.ny
    cartix = CartesianIndices((6,n_par.ny))
    row_ind = [linix[liqu[i,j],illiqu[i,j],cartix[i][2]] for i=1:6*n_par.ny, j=1:Ntot]
    naprob = 1 - m_par.λ  
    values = repeat([on_grid[1][2] .* naprob; (1 .- on_grid[1][2]) .* naprob; on_grid[2][2] .* on_grid[4][2] .* m_par.λ; (1 .- on_grid[2][2]) .* on_grid[4][2] .* m_par.λ; on_grid[3][2] .* (1 .- on_grid[4][2]) .* m_par.λ;(1 .- on_grid[3][2]) .* (1 .- on_grid[4][2]) .* m_par.λ],n_par.ny).*idios
    # nr_neg_weights = sum(values .< 0)
    # if(0 < nr_neg_weights)
    #     @warn("Negative weights obtained! Nr: ")
    #     print(nr_neg_weights)
    # end
    T = sparse(row_ind[:],repeat((1:Ntot)',6*n_par.ny)[:],values[:],Ntot,Ntot)
    dropzeros!(T)
    return T'
end