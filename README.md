# fast_poisson
Use finite difference method to solve Poisson equation in 1D,2D and 3D case. The matrix is solved with CG iteration and fast 
tensor product inversion. The result I found is that the fast tensor product inversion is fast than CG iterations in 2 and 3 dimensions. 
This is because the poisson matrix is sparse in those dimensions. 
