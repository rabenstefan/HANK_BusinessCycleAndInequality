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
                          BtoKminus::Real,
                          BtoK::Real,
                          rminus::Real,#::ForwardDiff.Dual{Nothing,Float64,5},
                          inc_lab::Array,#{Array{ForwardDiff.Dual{Nothing,Float64,5},3},1},
                          n_par::NumericalParameters,
                          m_par, meshes)

    ################### Copy/read-out stuff#####################################
#    @timev begin #0.008451 seconds (12 allocations: 7.178 MiB)
    β::Float64 = m_par.β
    # inc[1] = labor income , inc[2] = rental income,
    # inc[3]= capital liquidation income
    n = size(EVk)

    ############################################################################
    ## EGM Step 1: Find optimal liquid asset holdings in the constrained case ##
    ############################################################################
    EMU      = EVk .* β
    c_star = invmutil(EMU,m_par.ξ) # 6% of time with rolled out power function

    # Calculate assets consistent with choices being [k']
    # Calculate initial money position from the budget constraint
    # that leads to the optimal consumption choice
    k_star = (c_star .+ meshes.k*(Qminus + BtoK) .- inc_lab)./(Qminus + BtoKminus*RBminus/πminus+ rminus)

    # Next step: Interpolate w_guess and c_guess from new k-grids
    # using c[s,h,m"], m(s,h,m")
    # Interpolate grid().m and c_n_aux defined on m_star_n over grid().m
    # Policies for tuples (c*,m*,y) are now given. Need to interpolate to return to
    # fixed grid.
    c_interpol  = Array{eltype(c_star),2}(undef,(n[1],n[2]))#zeros(eltype(c_star_n), size(c_star_n)) # Initialize c_n-container
    for j in eachindex(n_par.grid_y)
        c_interpol[:,j] = mylinearinterpolate(k_star[:,j],c_star[:,j],n_par.grid_k)
    end
    # Check for binding borrowing constraints
    bc_binds = meshes.k .< repeat(k_star[1,:]',n_par.nk,1)
    # consumption, when tomorrow at borrowing constraint
    c_interpol[bc_binds] = meshes.k[bc_binds].*(Qminus + rminus + BtoKminus*RBminus/πminus) .+ inc_lab[bc_binds] .- n_par.grid_k[1]*(Qminus + BtoK)
    # need k_star as next period's optimal k-policy
    k_interpol = (meshes.k .* (Qminus + rminus + BtoKminus*RBminus/πminus) .+ inc_lab .- c_interpol) ./ (Qminus + BtoK)
    
    return c_interpol, k_interpol
end
