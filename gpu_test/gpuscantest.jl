using DelimitedFiles
using LinearAlgebra
using Optim
using Distributions

include("../gpu_src/gpuscan.jl")
# include("../gpu_src/kinship.jl")
include("../src/readData.jl")
include("../src/wls.jl")
include("../src/lmm.jl")


pheno_file = joinpath(@__DIR__, "..", "data", "bxdData", "traits.csv")
pheno = readBXDpheno(pheno_file)
geno_file = joinpath(@__DIR__, "..", "data", "bxdData", "geno_prob.csv")
geno = readGenoProb(geno_file)
# k = calcKinship(geno)
# getting kinship matrix from gemma 
kinship_file = joinpath(@__DIR__, "..", "test","output", "result.cXX.txt")
k = convert(Array{Float64,2},readdlm(kinship_file, '\t'))

# CPU scan
for i in 5#1:size(pheno)[2]
    # run_julia(pheno[:,i], geno, k, false, "alt")
    ## genome scan
    global (sigma2, h2, lod) = scan(reshape(pheno[:,i], :, 1), geno, k, false, "alt")
    ## genome scan permutation
    # scan(reshape(pheno[:,1], :, 1), geno, k, 1024,1,true);

    ## transform LOD to -log10(p) (univariate)
    # result = -log.(10,(ccdf.(Chisq(1),2*log(10)*lod)));
    # result = result[1:2:end]
    # return (result, sigma2, h2)
end

# GPU scan 




