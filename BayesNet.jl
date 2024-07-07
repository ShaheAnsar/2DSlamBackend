
import Pkg;
Pkg.activate(".venv");
using Random, Distributions;

mutable struct MultivariateGaussian
    μ::Vector{Float64} # Mean vector
    Σ::Matrix{Float64} # Covariance Matrix
    static::Bool # Tells us if we have to update the mean or not
    update_func::Function # Update function to use if we have to update the mean
    varlist::Vector{String} # The first string is the current variable, the rest are the dependents
end

function dist_update(g::MultivariateGaussian)
    if g.static # Don't upadte anything
        return
    else
        g.μ = update_func(g) # Update the mean
    end
end

function dist_sample(g::MultivariateGaussian, n::Int32 = 1)
    return rand(MvNormal(g.μ, g.Σ), n)
end

function dist_sample(g::MultivariateGaussian, varlist::Dict{String, Union{Float64,Vector{Float64}}},
        n::Int32 = 1)
    return 0
end

function dist_pdf(g::MultivariateGaussian, x::Vector{Float64})
    return pdf(g, x)
end

struct BayesNet
    nodes::Vector{String} # Name of variable. Index of variable is the index everywhere
    adj_list::Vector{Vector{Int32}} #Adjacency list, all nodes show their parents, children are not shown
    dists::Vector{MultivariateGaussian} # Distributions
    known_state::Dict{String, Union{Float64, Vector{Float64}}
end

function bayesnet_get_parentless_nodes(b::BayesNet)
    parentless_nodes = Vector{Tuple{String, Int32}}()
    for (i, list) in enumerate(adj_list) 
        if length(list) == 0
            push!(parentless_nodes, (b.nodes[i], i))
        end
    end
    return parentless_nodes
end

function bayesnet_ancestral_sampling(b::BayesNet)
    sampled = Set{Tuple{String, Int32}}
    nodes = bayesnet_get_parentless_nodes(b)
    while true
        for n in nodes
            sample = dist_sample(b.dists[n[2]])
            known_state[n[1]] = sample
            push!(sampled, n)
        end
    end
end

function bayesnet_joint_probability(b::BayesNet, state::Dict{String, Union{Float64, Vector{Float64}}})
    
end


struct BayesNetOptimizer
    net::BayesNet
    known_state::Dict{String, Union{Float64, Vector{Float64}}
end

# Sample test (2D)
#       z
#       ^
# x1 -> x2 -> x3
