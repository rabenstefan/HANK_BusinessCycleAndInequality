function FRAsys(X::AbstractArray, XPrime::AbstractArray, Xss::Array{Float64,1}, m_par,
    n_par::NumericalParameters, indexes)
    # The function call with Duals takes
    # Reserve space for error terms
    F = zeros(eltype(X),size(X))
    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(Xss[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(Xss[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

    @generate_equations(aggr_names)

    distrSS = ones(2)./2.0
    @include "../input_aggregate_model.jl"
    # change some rules: taxlev, T
    F[indexes.τlev] = log.(1.0 .- av_tax_rate) .- log.(τlev)
    F[indexes.T]    = log.(T) .- log.(av_tax_rate.*(w.*N .+ firm_profits))
    # C (with euler equ), K (state transition), B (from BtoK)
    # composite consumption X
    X = C - τlev.*(w.*N) * (1.0 - m_par.τ_prog)/(m_par.γ + 1.0)
    XPrime = CPrime - τlev.*(w.*N)*(1.0 - m_par.τ_prog)/(m_par.γ + 1.0)
    # write Rtot with liquid return RL --> general also for specification with stocks
    RtotPrime = (rPrime .+ qPrime .+ BtoKPrime.*RLPrime ./ πPrime)./(q .+ BtoKPrime)
    F[indexes.C] = log.(X) .- (log.(XPrime) .- log.(m_par.β .* RtotPrime)/m_par.ξ)
    F[indexes.Kstate] = log.(Y .- G .- I .+ (A .- 1.0) .* RB .* B ./ π .- (δ_1 * (u - 1.0) + δ_2 / 2.0 * (u - 1.0)^2.0).*K) .- log.(C)
    F[indexes.K] = log.(K) .- log.(Kstate)
    F[indexes.B] = log.(B) .- log.(BtoK.*K)
    # BtoK (no-arbitrage condition)
    F[indexes.BtoK] = log.(RLPrime./πPrime) .- (rPrime .+ qPrime)./q

    F[indexes.Ht] = log.(Ht) .- Xss[indexes.HtSS]
    return F
end