module FaSTLMM
# packages we need to work
using Optim
using DelimitedFiles
using Missings
using LinearAlgebra
using Distributed
using Random
using DataFrames
using Statistics
using Random



include("util.jl")

# code for (wls) weighted least squares
include("wls.jl")
export wls

# code for rorateData and flmm
include("lmm.jl")
# sata type we are exporting
export Flmm
# functions we are exporting
export rotateData, flmm

include("scan.jl")
export scan


include("bulkscan.jl")
export bulkscan
end # module
