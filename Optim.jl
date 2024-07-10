module Optim
using ForwardDiff;
using LinearAlgebra;

function newton_iter(x::Vector{Float64}, ∇::Vector{Float64}, H::Matrix{Float64})
    return x - inv(H) * ∇;
end

function newton(f::Function, d::Int64, iters=100)
    x = Vector{Float64}();
    for _d in range(1, d)
        push!(x, 0.0);
    end
    for iter in range(1, iters)
        x = newton_iter(x, ForwardDiff.gradient(f, x), ForwardDiff.hessian(f, x));
        println("iter: $iter, f(x)=$(f(x)), ∇=$(ForwardDiff.gradient(f, x))");
    end
    return x;
end

end
