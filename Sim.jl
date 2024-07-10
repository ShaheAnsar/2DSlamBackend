# Module that simulates the movement of a point robot in a 2D world.

module Sim
using Random;
mutable struct World
    positions::Vector{Vector{Float64}};
    landmarks::Vector{Vector{Float64}};
    odom::Vector{(Vector{Float64}, Float64)}; # (Position vector, theta)
    measurements::Vector{(Int64, Int64, Vector{Float64})}; # (Position index, landmark index, range-bearing measurement)
end

function generateWorld(p_num = 100, landmarks = 20, measurement_prob=0.7)
    w = World():
    positions = [];
    landmarks = [];
    positions = 
end

end
