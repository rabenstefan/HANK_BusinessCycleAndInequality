using JLD2
@load "hank_2asset_mcmc_1901_baseline_chain_all_commit.jld2" parnames
# Manually create par_final for HANK* with values from paper
vals = Dict(:δ_s=>1.420,:ϕ=>0.218,:κ=>0.105,:κw=>0.133,:ρ_R=>0.803,:σ_Rshock=>0.00266,:θ_π=>2.614,:θ_Y=>0.078,:γ_B=>0.157,:γ_π=>-1.175,:γ_Y=>-0.697,:ρ_Gshock=>0.992,:σ_Gshock=>0.00263,:ρ_τ=>0.552,:σ_Tlevshock=>0.00122,:γ_Bτ=>0.833,:γ_Yτ=>2.496,:ρ_P=>0.988,:σ_Tprogshock=>0.02014,:α_τ=>2.408,:ρ_A=>0.983,:σ_A=>0.00159,:ρ_Z=>0.992,:σ_Z=>0.00608,:ρ_ZI=>0.965,:σ_ZI=>0.02531,:ρ_μ=>0.9,:σ_μ=>0.01645,:ρ_μw=>0.909,:σ_μw=>0.05216,:ρ_s=>0.639,:σ_Sshock=>0.61443,:Σ_n=>0.634)
par_final = [vals[s] for s in parnames[1:33]]
par_final = [par_final;0.04076;0.03707;0.00038;0.03772;0.08074]
@save "mode_start_file_inequ.jld2" par_final parnames