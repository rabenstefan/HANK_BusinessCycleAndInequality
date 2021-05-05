@doc raw"""
    EGM_OneAssetpolicyupdate(EVm,EVk,Qminus,πminus,RBminus,Tshock,inc,n_par,m_par,warnme)

This function performs one backward iteration of the consumption policy
using the enogenous grid method. It requires the next period consumption
policy C_NEXT, this period's wages W, this periods interest rate R, and
next period's interest rate RPLUS aside model parameters as inputs.

# Returns
- `c_a_star`,`m_a_star`,`k_a_star`,`c_n_star`,`m_n_star`: optimal (on-grid) policies for
    consumption [`c`], liquid [`m`] and illiquid [`k`] asset, with [`a`] or
    without [`n`] adjustment of illiquid asset
"""
function EGM_OneAssetpolicyupdate(EVk::Array,#{ForwardDiff.Dual{Nothing,Float64,5},3},
                          Qminus::Real,#::ForwardDiff.Dual{Nothing,Float64,5},
                          πminus::Real,#::ForwardDiff.Dual{Nothing,Float64,5},
                          RBminus::Real,#::ForwardDiff.Dual{Nothing,Float64,5},
                          Tshock::Real,#::ForwardDiff.Dual{Nothing,Float64,5},
                          inc::Array,#{Array{ForwardDiff.Dual{Nothing,Float64,5},3},1},
                          n_par::NumericalParameters,
                          m_par, warnme)

    ################### Copy/read-out stuff#####################################
#    @timev begin #0.008451 seconds (12 allocations: 7.178 MiB)
    β::Float64 = m_par.β
    borrwedge = m_par.Rbar.*Tshock
    # inc[1] = labor income , inc[2] = rental income,
    # inc[3]= capital liquidation income
    inc_lab  = inc[1]
    inc_rent = inc[2]
    inc_IA   = inc[3]
    n = size(EVk)
    kmax     = n_par.grid_k[end]

    ############################################################################
    ## EGM Step 1: Find optimal liquid asset holdings in the constrained case ##
    ############################################################################
    EMU      = EVk .* β
    c_star = invmutil(EMU,m_par.ξ) # 6% of time with rolled out power function

    # Calculate assets consistent with choices being [k']
    # Calculate initial money position from the budget constraint
    # that leads to the optimal consumption choice
    k_star = (c_star .+ n_par.mesh_k*())
    m_star_n = (c_star_n .+ n_par.mesh_m .- inc_lab .- inc_rent)
    # Apply correct interest rate
    m_star_n  .= m_star_n./((RBminus .+ borrwedge.*(m_star_n.<0))./πminus)  # apply borrowing rate

    # Next step: Interpolate w_guess and c_guess from new k-grids
    # using c[s,h,m"], m(s,h,m")
    # Interpolate grid().m and c_n_aux defined on m_star_n over grid().m

    # Check monotonicity of m_star_n
    if warnme
        m_star_aux    = reshape(m_star_n,(n[1], n[2]*n[3]))
        if any(any(diff(m_star_aux, dims=1).<0))
            @warn "non monotone future liquid asset choice encountered"
#            display(find(diff(m_star_aux).<0))
        end
    end

    # Policies for tuples (c*,m*,y) are now given. Need to interpolate to return to
    # fixed grid.
    c_n_star  = Array{eltype(c_star_n),3}(undef,(n[1],n[2], n[3]))#zeros(eltype(c_star_n), size(c_star_n)) # Initialize c_n-container
    m_n_star  = Array{eltype(c_star_n),3}(undef,(n[1],n[2], n[3]))#zeros(eltype(c_star_n), size(c_star_n)) # Initialize m_n-container
    m_n_weights = Array{eltype(c_star_n),3}(undef,(n[1],n[2],n[3]))
    m_n_ind = Array{Int,3}(undef,(n[1],n[2],n[3]))
#end
#@timev begin #0.004493 seconds (4.48 k allocations: 5.332 MiB)
    @inbounds @views  begin
        for jj=1:n[3] # Loop over income states
            for kk = 1:n[2] # Loop over capital states
                cc, mn, lw, li = mylinearinterpolate_mult2(m_star_n[:,kk,jj], c_star_n[:,kk,jj],n_par.grid_m, n_par.grid_m)
                c_n_star[:,kk,jj]=cc
                m_n_star[:,kk,jj]=mn
                m_n_weights[:,kk,jj]=lw
                m_n_ind[:,kk,jj]=li
                # Check for binding borrowing constraints, no extrapolation from grid
                #begin
                    bcpol = m_star_n[1,kk,jj]
                    for mm= 1:n[1]
                       if n_par.mesh_m[mm,kk,jj] .<bcpol
                           c_n_star[mm,kk,jj] = inc_lab[mm,kk,jj] .+ inc_rent[mm,kk,jj] .+ inc_LA[mm,kk,jj] .- n_par.grid_m[1]
                           m_n_star[mm,kk,jj] = n_par.grid_m[1]
                           m_n_weights[mm,kk,jj] = 1
                           m_n_ind[mm,kk,jj] = 1
                       end
                       if mmax  .< m_n_star[mm,kk,jj]
                            m_n_star[mm,kk,jj] = mmax
                            m_n_weights[mm,kk,jj] = 0
                            m_n_ind[mm,kk,jj] = n[1]-1
                       end
                    end
                #end
            end
        end
    end
#end
#@timev begin #0.003771 seconds (2.25 k allocations: 3.795 MiB)
    #-------------------------END OF STEP 1-----------------------------

    ############################################################################
    ## EGM Step 2: Find Optimal Portfolio Combinations                        ##
    ############################################################################
    term1    = (β/Qminus) * EVk
    E_return_diff = term1 .- EMU

    # Find an m_a* for given k' that solves the difference equation [45]
    m_a_aux1 = Fastroot(n_par.grid_m,E_return_diff)  # Fastroot does not allow for extrapolation and uses non-negativity constraint and monotonicity
    m_a_aux  = reshape(m_a_aux1, (n[2],n[3]))
#end
#@timev begin #0.000675 seconds (375 allocations: 358.563 KiB)
    ###########################################################################
    ## EGM Step 3: Constraints for money and capital are not binding         ##
    ###########################################################################
    # Interpolation of psi()-function at m*_n[m,k]
    aux_index = (0:(n[2]*n[3])-1)*n[1]
    EMU_star = Array{eltype(m_a_aux),2}(undef,(n[2],n[3]))#zeros(eltype(m_a_aux),(n[2],n[3]))
    idx_saved = Array{Int,1}(undef,(n[2]*n[3]))
    s_saved = Array{eltype(m_a_aux),1}(undef,(n[2]*n[3]))
    step = diff(n_par.grid_m) #Stepsize on grid()
    # Interpolate EMU[m",k',s'*h',M',K'] over m*_n[k"], m-dim is dropped
    for j in eachindex(m_a_aux)
        xi = m_a_aux[j]
        if xi.> n_par.grid_m[n[1]-1]
            idx = n[1]-1
        elseif xi.< n_par.grid_m[1]
            idx = 1
        else
            idx = locate(xi,n_par.grid_m)  # find indexes on grid next smallest to optimal policy
        end
        s = (xi .- n_par.grid_m[idx])./step[idx] #Distance of optimal policy to next grid point
        EMU_star[j]        = EMU[idx .+ aux_index[j]].*(1.0 -s) .+
                              s.*(EMU[idx .+ aux_index[j].+1])#linear interpolation
        idx_saved[j] = idx
        s_saved[j] = s
    end
    c_a_aux        = invmutil(EMU_star, m_par.ξ)
    idx_saved = reshape(idx_saved,(n[2],n[3]))
    s_saved = reshape(s_saved,(n[2],n[3]))
    # Resources that lead to capital choice
    # k'= c + m*(k") + k" - w*h*N
    # = value of todays cap and money holdings
    Resource = c_a_aux .+ m_a_aux .+ inc_IA[1,:,:] .- inc_lab[1,:,:]

    # Money constraint is not binding, but capital constraint is binding
    m_star_zero = m_a_aux[1,:] # Money holdings that correspond to k'=0:  m*(k=0)

    # Use consumption at k"=0 from constrained problem, when m" is on grid()
    aux_c     = reshape(c_star_n[:,1,:],(n[1], n[3]))
    aux_inc   = reshape(inc_lab[1,1,:],(1, n[3]))
    cons_list = Array{Array{eltype(c_star_n)}}(undef,n[3],1)
    res_list  = Array{Array{eltype(c_star_n)}}(undef,n[3],1)
    mon_list  = Array{Array{eltype(c_star_n)}}(undef,n[3],1)
    cap_list  = Array{Array{eltype(c_star_n)}}(undef,n[3],1)

    sizes_kpr0 = zeros(Int,1 + n[1],n[3])
    for j=1:n[3]
        # When choosing zero capital holdings, HHs might still want to choose money
        # holdings smaller than m*(k'=0)
        if m_star_zero[j]>n_par.grid_m[1]
            # Calculate consumption policies, when HHs chooses money holdings
            # lower than m*(k"=0) and capital holdings k"=0 and save them in cons_list
            log_index    = n_par.grid_m.<m_star_zero[j]
            # aux_c is the consumption policy under no cap. adj. (fix k=0), for m<m_a*(k'=0)
            c_k_cons     = aux_c[log_index,j]
            cons_list[j] = c_k_cons; #Consumption at k"=0, m"<m_a*(0)
            # Required Resources: Money choice + Consumption - labor income
            # Resources that lead to k"=0 and m'<m*(k"=0)
            res_list[j] = n_par.grid_m[log_index] .+ c_k_cons .- aux_inc[j]
            mon_list[j] = n_par.grid_m[log_index]
            cap_list[j] = zeros(eltype(EVm),sum(log_index))
            sizes_kpr0[:,j] = [sum(log_index);log_index]
        else
            cons_list[j] = zeros(eltype(EVm),0) #Consumption at k"=0, m"<m_a*(0)
            # Required Resources: Money choice + Consumption - labor income
            # Resources that lead to k"=0 and m'<m*(k"=0)
            res_list[j] = zeros(eltype(EVm),0)
            mon_list[j] = zeros(eltype(EVm),0)
            cap_list[j] = zeros(eltype(EVm),0)
        end
    end

    # Merge lists
    c_a_aux  = reshape(c_a_aux,(n[2], n[3]))
    m_a_aux  = reshape(m_a_aux,(n[2], n[3]))

    for j=1:n[3]
        cons_list[j] = append!(cons_list[j], c_a_aux[:,j])
        res_list[j]  = append!(res_list[j],  Resource[:,j])
        mon_list[j]  = append!(mon_list[j],  m_a_aux[:,j])
        cap_list[j]  = append!(cap_list[j],  n_par.grid_k)
    end
#end
#@timev begin #0.001258 seconds (13 allocations: 4.790 MiB)
    ####################################################################
    ## EGM Step 4: Interpolate back to fixed grid                     ##
    ####################################################################
    c_a_star = Array{eltype(c_star_n),3}(undef,(n[1],n[2], n[3]))
    m_a_star = Array{eltype(c_star_n),3}(undef,(n[1],n[2], n[3]))#zeros(eltype(c_star_n), (n[1],n[2], n[3]))
    k_a_star = Array{eltype(c_star_n),3}(undef,(n[1],n[2], n[3]))#zeros(eltype(c_star_n), (n[1],n[2], n[3]))
    k_weights = Array{eltype(c_star_n),3}(undef,(n[1],n[2],n[3]))
    k_ind = Array{Int,3}(undef,(n[1],n[2],n[3]))
    m_a_weights = Array{eltype(c_star_n),3}(undef,(n[1],n[2],n[3]))
    m_a_weights_plus = Array{eltype(c_star_n),3}(undef,(n[1],n[2],n[3]))
    m_a_ind = Array{Int,3}(undef,(n[1],n[2],n[3]))
    m_a_ind_plus = Array{Int,3}(undef,(n[1],n[2],n[3]))
    Resource_grid  = reshape(inc_IA .+ inc_LA .+ inc_rent,(n[1].*n[2], n[3]))
    labor_inc_grid = inc_lab[1,1,:][:]#reshape(inc_lab,(n[1]*n[2], n[3]))
    log_index2     = zeros(Bool,n[1].*n[2])
    range_mgrid = (1:n[1])'
#end
#@timev begin #  0.009399 seconds (448 allocations: 4.878 MiB)
@views @inbounds begin
    for j=1:n[3]
        # Check monotonicity of resources
        if warnme
            if any(diff(res_list[j]).<0)
                @warn "non monotone resource list encountered"
            end
        end
        # when at most one constraint binds:
        log_index2[:] = reshape(Resource_grid[:,j],n[1]*n[2]).<res_list[j][1]
        # Lowest value of res_list corresponds to m_a"=0 and k_a"=0.
        c_a_star1, m_a_star1, k_a_star1, k_weights1, k_ind1 = mylinearinterpolate_mult3(res_list[j],cons_list[j],mon_list[j],cap_list[j],Resource_grid[:,j])
        m_a_weights1 = zeros(eltype(c_star_n),size(k_weights1))
        m_a_weights_plus1 = zeros(eltype(c_star_n),size(k_weights1))
        m_a_ind1 = zeros(Int,size(k_weights1))
        m_a_ind_plus1 = ones(Int,size(k_weights1))

        # Identify where index within 'k-grid' bounds, and set weights=1 otherwise.
        ind_kgrid = k_ind1 .> sizes_kpr0[1,j]
        k_weights1[Bool.(1 .- ind_kgrid)] .= 1
        m_a_weights1[Bool.(1 .- ind_kgrid)] .= 1
        # Exchange m_a_aux values in mon_list with indices on m-grid (interpolated above).
        mon_list_len = max(size(mon_list[j])...)
        mon_list1 = zeros(Int,mon_list_len)
        mon_list1[sizes_kpr0[1,j]+1:mon_list_len] = idx_saved[:,j]
        mon_list1[1:sizes_kpr0[1,j]] = range_mgrid[Bool.(sizes_kpr0[2:end,j])]
        m_a_ind1 = mon_list1[k_ind1]
        m_a_ind_plus1[ind_kgrid] = mon_list1[k_ind1[ind_kgrid] .+ 1]
        mon_list[j] = mon_list1
        mon_list[j][sizes_kpr0[1,j]+1:mon_list_len] = 1 .- s_saved[:,j]
        m_a_weights1[ind_kgrid] = mon_list[j][k_ind1[ind_kgrid]]
        m_a_weights_plus1[ind_kgrid] = mon_list[j][k_ind1[ind_kgrid] .+ 1]
        # Fix k_ind1.
        k_ind1[ind_kgrid] = k_ind1[ind_kgrid] .- sizes_kpr0[1,j]
        k_ind1[Bool.(1 .- ind_kgrid)] .= 1;

        # Any resources on grid smaller then res_list imply that HHs consume all
        # resources plus income.
        # When both constraints are binding:
         c_a_star1[log_index2]  = Resource_grid[log_index2,j] .+ labor_inc_grid[j] .- n_par.grid_m[1]
         m_a_star1[log_index2] .= n_par.grid_m[1]
         k_a_star1[log_index2] .= 0.0
         k_weights1[log_index2] .= 1
         k_ind1[log_index2] .= 1
         m_a_ind1[log_index2] .= 1
         m_a_weights1[log_index2] .= 1
         for kk = 1:n[2]
            for mm = 1:n[1]
                runind = mm + (kk-1)*n[1]
                mp = m_a_star1[runind]
                kp = k_a_star1[runind]
                kw = k_weights1[runind]
                ki = k_ind1[runind]
                c_a_star[mm,kk,j] = c_a_star1[runind]
                m_a_weights[mm,kk,j] = m_a_weights1[runind]
                m_a_weights_plus[mm,kk,j] = m_a_weights_plus1[runind]
                m_a_ind[mm,kk,j] = m_a_ind1[runind]
                m_a_ind_plus[mm,kk,j] = m_a_ind_plus1[runind]
                if mp <mmax
                    m_a_star[mm,kk,j] = mp
                else
                    m_a_star[mm,kk,j] = mmax
                end
                if kp < kmax
                    k_a_star[mm,kk,j] = kp
                    k_weights[mm,kk,j] = kw
                    k_ind[mm,kk,j] = ki
                else
                    k_a_star[mm,kk,j] = kmax
                    k_weights[mm,kk,j] = 0
                    k_ind[mm,kk,j] = n[2] - 1
                end
            end
        end
    end
end
N = n[1]*n[2]*n[3]

m_n_ind = reshape(m_n_ind,1,N)
m_n_weights = reshape(m_n_weights,1,N)
k_ind= reshape(k_ind,1,N)
k_weights = reshape(k_weights,1,N)
m_a_ind= reshape(m_a_ind,1,N)
m_a_weights = reshape(m_a_weights,1,N)
m_a_ind_plus = reshape(m_a_ind_plus,1,N)
m_a_weights_plus = reshape(m_a_weights_plus,1,N)
on_grid = ((m_n_ind,m_n_weights),(m_a_ind,m_a_weights),(m_a_ind_plus,m_a_weights_plus),(k_ind,k_weights))
    return c_a_star, m_a_star, k_a_star, c_n_star, m_n_star, on_grid
end
