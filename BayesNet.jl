
import Pkg;
Pkg.activate(".venv");
using Random, Distributions, LinearAlgebra;
include("./Optim.jl");
import .Optim;

@enum NodeType::Int32   Position MeasBearing MeasOdom MeasRangeBearing MeasLoopClosure Landmark
@enum MotionModelType::Int32 ConstVel ConstAcc PrevVel Odom
@enum AdjIndex::Int32 HANDLE_INDEX=1 SIGMA_INDEX=2

# Assume 2D for everything that follows

mutable struct MvGaussian{T <: Real}
    Σ::Matrix{T}
    initialized::Bool
end

mutable struct SimpleMvGaussian{T <: Real}
    σ²::T# We only use one float to define the variance. All covariances are 0, so Σ will be a σ²*I
    initialized::Bool
end

mutable struct Node
    name::String; # Unique name
    handle::Int32; # Handle used to index it in the adjacency list
    value::Vector{Float64};
    type::NodeType;
    dist::Union{MvGaussian, SimpleMvGaussian};
    params::Any;
end

mutable struct BayesNet
    state::Dict{String, Node};
    variables::Vector{String};
    adj_list::Vector{Vector{(Int32)}}; # Unlike classic graphs,
                                                # here the list for every node will list its parents
    motion_model::MotionModelType;
    motion_model_info::Vector{Float64}; # Contains the velocity for ConstVel, acc followed by vel for ConstAcc
                                        # Ignore PrevVel and Odom for the time
    node_counter::Int32;
end


function BayesNet_create(filename::String)
    return nothing; # We'll come to this later
end

function BayesNet_add_node(b::BayesNet, name::String, value::Vector{Float64}, t::NodeType, p::Any)
    push!(b.variables, name);
    b.state[name] = Node(name, b.node_counter, value, t, SimpleMvGaussian(1.0, false), p);
    b.node_counter += 1;
end

# Use the below function to construct the adjacency list
function BayesNet_construct(b::BayesNet)
    for i in range(1, b.node_counter - 1)
        println("Iter: ", i);
        push!(b.adj_list, Vector{Float64}());
    end
end

# Once constructed, populate the adjacency list
function BayesNet_add_edge(b::BayesNet, p::String, c::String)
    push!(b.adj_list[b.state[c].handle], b.state[p].handle)
end


function BayesNet_init_dists(b::BayesNet)
end

module Measurement
function bearing(dir::Vector{T}) where T <: Real
    return atan(dir[2], dir[1])
end

function rangebearing(dir::Vector{T}) where T <: Real
    return [ sqrt(dir[1]^2 + dir[2]^2), atan(dir[2], dir[1]) ]
end

function odom(xnew::Vector{T}, xold::Vector{T}) where T <: Real
    return xnew - xold;
end
end

function BayesNet_measure_func(b::BayesNet, t::NodeType)
    if t == "Position" || t == "Landmark"
        println("You fucked up kiddo");
    elseif t == "MeasBearing"
    elseif t == "MeasOdom"
    elseif t == "MeasRangeBearing"
    end
end

module MotionModel
function constant_vel(xold::Vector{T}, v::Vector{Float64}, dt::Float64) where T
    return xold + v*dt;
end

function constant_acc(xold::Vector{Float64}, a::Vector{Float64}, v::Vector{Float64}, dt::Float64)
    return xold + v*dt + 0.5*a*dt^2;
end
end

function NodePdf(g::MvGaussian{T1}, μ::Vector{T2}, x::Vector{T3}) where {T1 <: Real, T2 <: Real, T3 <: Real}
    Σ = g.Σ;
    δx = x - μ;
    return 1.0/(2π*sqrt(det(Σ))) * exp(-0.5 * transpose(δx)*inv(Σ)*δx)
end

function NodePdf(g::SimpleMvGaussian{T1}, μ::Vector{T2}, x::Vector{T3}) where {T1 <: Real, T2 <: Real, T3 <: Real}
    σ² = g.σ²;
    δx = x - μ;
    Σ = [σ² 0; 0 σ²];
    return 1.0/(2π*sqrt(det(Σ))) * exp(-0.5 * transpose(δx)*inv(Σ)*δx)
end

function NodeLsqFunc(g::SimpleMvGaussian{T1}, μ::Vector{T2}, x::Vector{T3}) where {T1 <: Real, T2 <: Real, T3 <:Real}
    σ² = g.σ²;
    δx = x - μ;
    Σ = [σ² 0; 0 σ²];
    return transpose(δx) * inv(Σ) * δx;
end

function NodeLsqFunc(g::MvGaussian{T1}, μ::Vector{T2}, x::Vector{T3}) where {T1 <: Real, T2 <: Real, T3 <: Real}
    Σ = g.Σ;
    return transpose(δx) * inv(Σ) * δx;
end

#function NodeUpdateMean(g::Union{SimpleMvGaussian, MvGaussian}, x::Vector{T}) where T
#    g.μ = x;
#end

function BayesNet_compute_joint_probability(b::BayesNet)
    joint_probability = 1.0
    computed = Set{Int32}()
    while length(computed) != length(b.adj_list)
        for i in range(1, b.node_counter - 1)
            computable = true
            if i ∈ computed
                continue
            else
                for n_p in b.adj_list[i]
                    if n_p ∉ computed
                        computable = false
                    end
                end
            end
            if computable
                name = b.variables[i]
                println("$name is computable!");
                state = b.state[name];
                t = state.type;
                if t == Position
                    if length(b.adj_list[i]) == 0
                        println("Found parentless position node");
                        joint_probability *= 1.0; # Act as if this node affects nothing
                    elseif length(b.adj_list[i]) != 1 # There should be no more than one parent to a position node
                        println("You fucked up with the position node kiddo");
                    elseif b.motion_model == ConstVel
                        # Calculate the position for the current node from the parent node
                        xold = b.state[b.variables[b.adj_list[i][1]]].value;
                        xnew = MotionModel.constant_vel(xold,
                                                        b.motion_model_info, 1.0);
                        println("Old Position: $xold, New Position: $( state.value ), Expected Position: $xnew");
                        # Update the current mean in the distribution we have
                        # Calculate the joint probability
                        contrib = NodePdf(state.dist, xnew, state.value);
                        println("Contributing Factor: $contrib");
                        joint_probability *= contrib
                    elseif b.motion_model == ConstAcc
                        println("Get everything else working doofus");
                    elseif b.motion_model == PrevVel
                        println("Get everything else working doofus");
                    elseif b.motion_model == Odom
                        println("Get everything else working doofus");
                    end
                elseif t == MeasBearing
                    println("Get everything else working doofus");
                elseif t == MeasOdom
                    println("Get everything else working doofus");
                elseif t == MeasRangeBearing
                    println("Got RangeBearing");
                    # Measurements of this kind will
                    # depend on one landmark and one position
                    # We will first get the expected range bearing
                    # We will then use the measurement to get the probability

                    # If there are more than 2 parents for this node, we fucked up
                    if length(b.adj_list[i]) != 2
                        println("Measurement RangeBearing fucked");
                    end
                    pnode::Union{Node, Nothing} = nothing;
                    lnode::Union{Node, Nothing} = nothing;
                    _handle = b.adj_list[i][1]
                    _name = b.variables[_handle];
                    _node = b.state[_name]
                    if _node.type == Position
                        pnode = _node;
                        lnode = b.state[b.variables[b.adj_list[i][2]]];
                    elseif _node.type == Landmark
                        lnode = _node;
                        pnode = b.state[b.variables[b.adj_list[i][2]]];
                    else
                        println("Something is fucked with the MeasRangeBearing parents!!");
                    end

                    μ = Measurement.rangebearing(lnode.value - pnode.value);
                    println("Measurement: $( state.value ), Expected Measurement: $μ");
                    contrib = NodePdf(state.dist, μ, state.value);
                    println("Contributing Factor: $contrib");
                    joint_probability *= contrib;
                elseif t == Landmark
                    println("Landmark node found! Ignoring...");
                end
                push!(computed, i)
            end
        end
    end
    return joint_probability;
end

# Function to optimize over with LSQ methods
function BayesNet_compute_lsq(b::BayesNet)
    lsq_sum = 0.0
    computed = Set{Int32}()
    while length(computed) != length(b.adj_list)
        for i in range(1, b.node_counter - 1)
            computable = true
            if i ∈ computed
                continue
            else
                for n_p in b.adj_list[i]
                    if n_p ∉ computed
                        computable = false
                    end
                end
            end
            if computable
                name = b.variables[i]
                println("$name is computable!");
                state = b.state[name];
                t = state.type;
                if t == Position
                    if length(b.adj_list[i]) == 0
                        println("Found parentless position node");
                        lsq_sum += 0.0; # Act as if this node affects nothing
                    elseif length(b.adj_list[i]) != 1 # There should be no more than one parent to a position node
                        println("You fucked up with the position node kiddo");
                    elseif b.motion_model == ConstVel
                        # Calculate the position for the current node from the parent node
                        xold = b.state[b.variables[b.adj_list[i][1]]].value;
                        xnew = MotionModel.constant_vel(xold,
                                                        b.motion_model_info, 1.0);
                        println("Old Position: $xold, New Position: $( state.value ), Expected Position: $xnew");
                        # Update the current mean in the distribution we have
                        # Calculate the joint probability
                        contrib = NodeLsqFunc(state.dist, xnew, state.value);
                        println("Contributing Factor: $contrib");
                        lsq_sum += contrib
                    elseif b.motion_model == ConstAcc
                        println("Get everything else working doofus");
                    elseif b.motion_model == PrevVel
                        println("Get everything else working doofus");
                    elseif b.motion_model == Odom
                        println("Get everything else working doofus");
                    end
                elseif t == MeasBearing
                    println("Get everything else working doofus");
                elseif t == MeasOdom
                    println("Get everything else working doofus");
                elseif t == MeasRangeBearing
                    println("Got RangeBearing");
                    # Measurements of this kind will
                    # depend on one landmark and one position
                    # We will first get the expected range bearing
                    # We will then use the measurement to get the probability

                    # If there are more than 2 parents for this node, we fucked up
                    if length(b.adj_list[i]) != 2
                        println("Measurement RangeBearing fucked");
                    end
                    pnode::Union{Node, Nothing} = nothing;
                    lnode::Union{Node, Nothing} = nothing;
                    _handle = b.adj_list[i][1]
                    _name = b.variables[_handle];
                    _node = b.state[_name]
                    if _node.type == Position
                        pnode = _node;
                        lnode = b.state[b.variables[b.adj_list[i][2]]];
                    elseif _node.type == Landmark
                        lnode = _node;
                        pnode = b.state[b.variables[b.adj_list[i][2]]];
                    else
                        println("Something is fucked with the MeasRangeBearing parents!!");
                    end

                    μ = Measurement.rangebearing(lnode.value - pnode.value);
                    println("Measurement: $( state.value ), Expected Measurement: $μ");
                    contrib = NodeLsqFunc(state.dist, μ, state.value);
                    println("Contributing Factor: $contrib");
                    lsq_sum += contrib;
                elseif t == Landmark
                    println("Landmark node found! Ignoring...");
                end
                push!(computed, i)
            end
        end
    end
    return lsq_sum; 
end

function BayesNet_compute_lsq(b::BayesNet, m::Dict{String, Any})
    lsq_sum = 0.0
    computed = Set{Int32}()
    while length(computed) != length(b.adj_list)
        for i in range(1, b.node_counter - 1)
            computable = true
            if i ∈ computed
                continue
            else
                for n_p in b.adj_list[i]
                    if n_p ∉ computed
                        computable = false
                    end
                end
            end
            if computable
                name = b.variables[i]
                println("$name is computable!");
                state = b.state[name];
                t = state.type;
                if t == Position
                    if length(b.adj_list[i]) == 0
                        println("Found parentless position node");
                        lsq_sum += 0.0; # Act as if this node affects nothing
                    elseif length(b.adj_list[i]) != 1 # There should be no more than one parent to a position node
                        println("You fucked up with the position node kiddo");
                    elseif b.motion_model == ConstVel
                        # Calculate the position for the current node from the parent node
                        xold = m[b.variables[b.adj_list[i][1]]];
                        xnew = MotionModel.constant_vel(xold,
                                                        b.motion_model_info, 1.0);
                        println("Old Position: $xold, New Position: $(m[name]), Expected Position: $xnew");
                        # Update the current mean in the distribution we have
                        # Calculate the joint probability
                        contrib = NodeLsqFunc(state.dist, xnew, m[name]);
                        println("Contributing Factor: $contrib");
                        lsq_sum += contrib
                    elseif b.motion_model == ConstAcc
                        println("Get everything else working doofus");
                    elseif b.motion_model == PrevVel
                        println("Get everything else working doofus");
                    elseif b.motion_model == Odom
                        println("Get everything else working doofus");
                    end
                elseif t == MeasBearing
                    println("Get everything else working doofus");
                elseif t == MeasOdom
                    println("Get everything else working doofus");
                elseif t == MeasRangeBearing
                    println("Got RangeBearing");
                    # Measurements of this kind will
                    # depend on one landmark and one position
                    # We will first get the expected range bearing
                    # We will then use the measurement to get the probability

                    # If there are more than 2 parents for this node, we fucked up
                    if length(b.adj_list[i]) != 2
                        println("Measurement RangeBearing fucked");
                    end
                    pnode::Union{Node, Nothing} = nothing;
                    lnode::Union{Node, Nothing} = nothing;
                    lnode_value::Any = nothing;
                    pnode_value::Any = nothing;
                    _handle = b.adj_list[i][1]
                    _name = b.variables[_handle];
                    _node = b.state[_name]
                    if _node.type == Position
                        pnode = _node;
                        pnode_value = m[_name];
                        lnode = b.state[b.variables[b.adj_list[i][2]]];
                        lnode_value = m[b.variables[b.adj_list[i][2]]];
                    elseif _node.type == Landmark
                        lnode = _node;
                        lnode_value = m[_name];
                        pnode = b.state[b.variables[b.adj_list[i][2]]];
                        pnode_value = m[b.variables[b.adj_list[i][2]]];
                    else
                        println("Something is fucked with the MeasRangeBearing parents!!");
                    end

                    μ = Measurement.rangebearing(lnode_value - pnode_value);
                    z = m[name];
                    println("Measurement: $(m[name]), Expected Measurement: $μ");
                    contrib = NodeLsqFunc(state.dist, μ, z);
                    println("Contributing Factor: $contrib");
                    lsq_sum += contrib;
                elseif t == Landmark
                    println("Landmark node found! Ignoring...");
                end
                push!(computed, i)
            end
        end
    end
    return lsq_sum; 
end
b = BayesNet(Dict{String, Node}(), Vector{String}(), Vector{Vector{Int32}}(), ConstVel, [1.0, 0.0], 1);
BayesNet_add_node(b, "x1", [0.0, 0.], Position, nothing);
BayesNet_add_node(b, "x2", [1., 0.], Position, nothing);
BayesNet_add_node(b, "x3", [2., 0.], Position, nothing);
BayesNet_add_node(b, "z1", [13.153, 0.1526] + randn(2)./[5, 300], MeasRangeBearing, nothing);
BayesNet_add_node(b, "z2", [12.1655, 0.16514] + randn(2)./[5, 300], MeasRangeBearing, nothing);
BayesNet_add_node(b, "z3", [16.97, 0.7853] + randn(2)./[5, 300], MeasRangeBearing, nothing);
#BayesNet_add_node(b, "z1", [13.153, 0.1526], MeasRangeBearing, nothing);
#BayesNet_add_node(b, "z2", [12.1655, 0.16514], MeasRangeBearing, nothing);
#BayesNet_add_node(b, "z3", [16.97, 0.7853], MeasRangeBearing, nothing);
BayesNet_add_node(b, "l1", [13., 2.], Landmark, nothing);
BayesNet_add_node(b, "l2", [14., 12.], Landmark, nothing);
BayesNet_construct(b);

BayesNet_add_edge(b, "x1", "x2");
BayesNet_add_edge(b, "x2", "x3");
BayesNet_add_edge(b, "x1", "z1");
BayesNet_add_edge(b, "l1", "z1");
BayesNet_add_edge(b, "x2", "z2");
BayesNet_add_edge(b, "l1", "z2");
BayesNet_add_edge(b, "x3", "z3");
BayesNet_add_edge(b, "l2", "z3");

println(BayesNet_compute_joint_probability(b));
println(BayesNet_compute_lsq(b));

function lsq_optim_fn(b)
    return function(x::Vector)
        if length(x) != 6
            print("boii make sure length is 6!");
        end
        println("mapping ver");
        mapping = Dict{String, Any}()
        for (k,v) in b.state
            mapping[k] = v.value;
        end
        mapping["x1"] = x[1:2];
        mapping["x2"] = x[3:4];
        mapping["x3"] = x[5:6];
        return BayesNet_compute_lsq(b, mapping);
    end
end

function f2(x::Vector)
    return x'x;
end

f = lsq_optim_fn(b);
x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
import ForwardDiff;
ForwardDiff.gradient(f, x);
X_MAP = Optim.newton(f, 6);
println("X MAP is $X_MAP");
# Sample test (2D)
#       z
#       ^
# x1 -> x2 -> x3
