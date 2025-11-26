using StaticArrays
using LinearAlgebra

L_op = SMatrix{2,2,ComplexF64}([0 1; 0 0])
term = Matrix(L_op' * L_op)
println("L_op' * L_op = ", term)

v = [0.0, 1.0]
res = v' * term * v
println("v' * term * v = ", res)

