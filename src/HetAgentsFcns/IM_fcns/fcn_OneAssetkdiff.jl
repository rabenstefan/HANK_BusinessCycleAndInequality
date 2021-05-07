@doc raw"""
    Kdiff(K_guess,n_par,m_par)

Calculate the difference between the capital stock that is assumed and the capital
stock that prevails under that guessed capital stock's implied prices when
households face idiosyncratic income risk (Aiyagari model).

Requires global functions `employment(K,A,m_par)`, `interest(K,A,N,m_par)`,
`wage(K,A,N,m_par)`, `output(K,A,N,m_par)`, `profitsSS_fnc(Y,m_par)`, and [`Ksupply()`](@ref).

# Arguments
- `K_guess::Float64`: capital stock guess
- `n_par::NumericalParameters`, `m_par::ModelParameters`
"""
function OneAssetKdiff(K_guess::Float64,BtoK, n_par::NumericalParameters, m_par)
    N       = employment(K_guess, 1.0 ./(m_par.μ*m_par.μw), m_par)
    r       = interest(K_guess,1.0 ./m_par.μ,N, m_par)
    w       = wage(K_guess,1 ./m_par.μ,N, m_par)
    Y = output(K_guess,1.0,N,m_par)
    profits = profitsSS_fnc(Y,m_par.RB,m_par)
    K::Float64   = OneAssetKsupply(m_par.RB,1.0+r,w*N/n_par.H,profits,BtoK,n_par,m_par)[1]
    diff    = K - K_guess
    return diff
end