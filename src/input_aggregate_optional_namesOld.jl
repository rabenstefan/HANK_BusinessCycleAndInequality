# Collect essential aggregate states and controls, i.e. those that are needed in heterogeneous agent part.

const state_names_ess=["A","RB","σ"]

# Note: when concatenating with other controls, make sure `profits` stays last element in array.
const control_names_ess=["r","w","K","π","q","N","mcw","Ht","av_tax_rate","T","B","τlev","τprog","GiniW", "GiniC", "GiniX", "GiniI", "sdlgC", "P9010C", "I90share", "I90sharenet","P9010I", "w90share", "P10C", "P50C", "P90C","profits"]
const aggr_names_ess = [state_names_ess;control_names_ess]

const state_names_opt=["Z", "ZI", "μ","μw", "Ylag", "Blag", "Zlag", "Tlag",
"Glag", "Ilag", "wlag", "qlag", "Nlag", "Clag", "πlag", "σlag", "rlag", "RBlag",
"av_tax_ratelag", "τproglag", "mcwwlag", "Gshock", "Tlevshock", "Tprogshock", "Rshock", "Sshock"]

const control_names_opt=["πw", "Y" ,"C", "mc", "u","I","BY","TY", "mcww","G","τprog_obs","Ygrowth", "Bgrowth", "Zgrowth",
"Ggrowth", "Igrowth", "wgrowth", "qgrowth", "Ngrowth", "Cgrowth", "πgrowth", "σgrowth",
"τproggrowth", "rgrowth", "RBgrowth", "mcwwgrowth", "Tgrowth"]