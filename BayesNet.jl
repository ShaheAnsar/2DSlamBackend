using Random, Distributions;

import Pkg;
Pkg.activate(".venv");

struct MultivariateGaussian
    μ::Vector{Float64} # Mean vector
    Σ::Matrix{Float64} # Covariance Matrix
end

function gaussian_sample(g::MultivariateGaussian)
end

struct BayesNet
    nodes::Vector{Tuple{String, Int32}} #Name and integer pairs
    adj_list::Vector{Vector{Int32}} #Adjacency list
    gen_functions::Vector{Any} # Generative functions
end


struct BayesNetOptimizer
    net::BayesNet
    known_state::Vector{Tuple{String, Float64}}
end

# Sample test (2D)
#       z
#       ^
# x1 -> x2 -> x3
