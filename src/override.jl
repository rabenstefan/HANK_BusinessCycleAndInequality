push!(LOAD_PATH,"../../")
using HANKEstim, JLD2



function Fsys_agg(X::AbstractArray, XPrime::AbstractArray, Xss::Array{Float64,1},distrSS::AbstractArray, m_par::ModelParameters,
    n_par::NumericalParameters, indexes::Union{IndexStructAggr,IndexStruct})
    return zeros(eltype(X),size(X))
end

sr = load_steadystate()

@load "linearresults.jld2"
gx,hx,alarm_sgu,nk,A,B = SGU_estim(sr.XSSaggr,lr.A,lr.B,sr.m_par,sr.n_par,sr.indexes,sr.indexes_aggr,sr.distrSS;estim=true,Fsys_agg=Fsys_agg)