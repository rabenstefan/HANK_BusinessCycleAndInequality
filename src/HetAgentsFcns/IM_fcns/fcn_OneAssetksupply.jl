@doc raw"""
    Ksupply(RB_guess,R_guess,w_guess,profit_guess,n_par,m_par)

Calculate the aggregate savings when households face idiosyncratic income risk.

Idiosyncratic state is tuple ``(m,k,y)``, where
``m``: liquid assets, ``k``: illiquid assets, ``y``: labor income

# Arguments
- `R_guess`: real interest rate illiquid assets
- `RL_guess`: return on liquid assets
- `w_guess`: wages
- `profit_guess`: profits (payout to entrepreneurs)
- `n_par::NumericalParameters`
- `m_par::ModelParameters`

# Returns
- `K`,`B`: aggregate saving in illiquid (`K`) and liquid (`B`) assets
-  `TransitionMat`,`TransitionMat_a`,`TransitionMat_n`: `sparse` transition matrices
    (average, with [`a`] or without [`n`] adjustment of illiquid asset)
- `distr`: ergodic steady state of `TransitionMat`
- `c_a_star`,`m_a_star`,`k_a_star`,`c_n_star`,`m_n_star`: optimal policies for
    consumption [`c`], liquid [`m`] and illiquid [`k`] asset, with [`a`] or
    without [`n`] adjustment of illiquid asset
- `V_m`,`V_k`: marginal value functions
"""
function OneAssetKsupply(RL_guess::Float64,R_guess::Float64, w_guess::Float64,profit_guess::Float64,BtoK, n_par::NumericalParameters, m_par)
    #----------------------------------------------------------------------------
    # Initialize policy function guess
    #----------------------------------------------------------------------------
    # inc[1] = labor income , inc[2] = rental income,
    # inc[3]= liquid assets income, inc[4] = capital liquidation income
    meshes = (k = repeat(n_par.grid_k,1,n_par.ny), y = repeat(n_par.grid_y',n_par.nk,1))
    H       = n_par.H
    Paux    = n_par.Π^1000
    distr_y = Paux[1,:]

    inc_lab = Array{Float64,2}(undef,n_par.nk,n_par.ny)
    mcw = 1.0 ./ m_par.μw
    entr_laborinc = m_par.y_e .* mcw .* w_guess

    # labor income
    incgross = n_par.grid_y .* mcw.*w_guess
    incgross[end]= n_par.grid_y[end]*profit_guess .+ entr_laborinc
    incnet   = m_par.τ_lev.*(mcw.*w_guess.*n_par.grid_y).^(1.0-m_par.τ_prog)
    incnet[end]= m_par.τ_lev.*((n_par.grid_y[end] .* profit_guess).^(1.0-m_par.τ_prog) .+ (entr_laborinc).^(1.0 .- m_par.τ_prog))
    av_tax_rate = dot((incgross - incnet),distr_y)./dot((incgross),distr_y)

    GHHFA=((m_par.γ - m_par.τ_prog)/(m_par.γ+1)) # transformation (scaling) for composite good
    inc_lab = GHHFA.*m_par.τ_lev.*(meshes.y.*mcw.*w_guess).^(1.0-m_par.τ_prog) .+
             (1.0 .- mcw).*w_guess*n_par.H.*(1.0 .- av_tax_rate)# labor income net of taxes
    inc_lab[:,end]= m_par.τ_lev.*((meshes.y[:,end]*profit_guess).^(1.0-m_par.τ_prog) .+ GHHFA .* (entr_laborinc).^(1.0 .- m_par.τ_prog)) # profit income net of taxes

    q       = 1.0 # price of Capital
    π       = m_par.π # inflation (gross)
    c_guess = inc_lab .+ meshes.k*(R_guess - 1 + q +BtoK*RL_guess/π)
    if any(any(c_guess.<0))
        @warn "negative consumption guess"
    end
    dist    = 9999.0
    
    #----------------------------------------------------------------------------
    # Iterate over consumption policies
    #----------------------------------------------------------------------------
    count = 0
    n        = size(c_guess)
    c_star = zeros(n)
    k_star = zeros(n)
    N = n[1]*n[2]
    asset_ret = (R_guess-1.0+BtoK+q)/(q+BtoK)
    Vk      =  asset_ret.* mutil(c_guess,m_par.ξ)

    while dist > n_par.ϵ # Iterate consumption policies until converegence
        count = count + 1
        # Take expectations for labor income change
        EVk = Vk*n_par.Π'

        # Policy update step
        c_star, k_star = EGM_OneAssetpolicyupdate(EVk,q,m_par.π,RL_guess,BtoK,BtoK,R_guess - 1.0,inc_lab,n_par,m_par, meshes)

        Vk_new = asset_ret .* mutil(c_star,m_par.ξ)

        # Calculate distance in updates
        dist  = maximum(abs, invmutil(Vk_new,m_par.ξ) .- invmutil(Vk,m_par.ξ))
        Vk    = Vk_new
    end

    #------------------------------------------------------
    # Find stationary distribution (Is direct transition better for large model?)
    #------------------------------------------------------


    # Define transition matrix
    # TransitionMat = build_transition_matrix(on_grid,n_par,m_par)
    S, T, W = MakeTransition(k_star,n_par.Π, n_par)
    TransitionMat   = sparse(S,T,W, n_par.nk * n_par.ny, n_par.nk * n_par.ny)
    # if n_par.ny>8
    #     # Direct Transition
    #     distr = n_par.dist_guess #ones(n_par.nm, n_par.nk, n_par.ny)/(n_par.nm*n_par.nk*n_par.ny)
    #     distr, dist, count = MultipleDirectTransition(m_a_star, m_n_star, k_a_star, distr, m_par.λ, n_par.Π, n_par)
    # else
        # Calculate left-hand unit eigenvector (seems slow!!!)
    aux = real.(eigsolve(TransitionMat', 1)[2][1])
    # # Use GMRES (Krylov iteration) with incomplete LU-preconditioning to solve for x: Ax = 0
    # A = TransitionMat' - sparse(I,N,N)
    # LU = ilu(A,τ=0.001)
    # # starting guess
    # x = fill(1/N,N)
    # gmres!(x,A,zeros(N),Pl=LU)
    #     aux::Array{Float64,2} = eigs(TransitionMat';nev=1,sigma=1)[2]
    #     if isreal(aux)
    #         aux = real(aux)
    #     else
    #         error("complex eigenvector of transition matrix")
    #     end
    distr = reshape((aux[:])./sum((aux[:])),  (n_par.nk, n_par.ny))
    # end
    #-----------------------------------------------------------------------------
    # Calculate capital stock
    #-----------------------------------------------------------------------------
    K = sum(distr' * n_par.grid_k)
    return K, TransitionMat, distr, c_star, k_star, Vk
end
