import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import time

class OneDimSolver:
    def initializeGrid(self, xmin, xmax, npoints):
        self.xmin = xmin
        self.xmax = xmax
        self.npoints = npoints
        self.npoints_wbc = npoints-2
        self.delta_x = (self.xmax - self.xmin)/(self.npoints - 1)
        self.xgrid = np.linspace(xmin, xmax, npoints)
        self.phi = np.zeros(self.npoints)
        self.fixed_pot_grid = np.zeros(self.npoints)

    def initializeBoundaryCondition(self, bctype, phiminval, phimaxval):
        self.phiminval = phiminval
        self.phimaxval = phimaxval
        self.bctype = bctype
        if (bctype == 0): # Dirichlet
            self.defineDirichletBCVector1D()
        elif (bctype == 1): # Left Neumann Right Dirichlet
            self.defineNeumannBCVectorLeft()
        elif (bctype == 2): # Left Dirichlet Right Neumann
            self.defineNeumannBCVectorRight()
    
    def defineDirichletBCVector1D(self):
        self.bcvec = np.zeros(self.npoints_wbc)
        self.bcvec[0] = self.phiminval
        self.bcvec[-1] = self.phimaxval
    
    def defineNeumannBCVectorLeft(self):
        self.bcvec = np.zeros(self.npoints_wbc)
        self.bcvec[0] = -self.delta_x * self.phiminval
        self.bcvec[-1] = self.phimaxval
    
    def defineNeumannBCVectorRight(self):
        self.bcvec = np.zeros(self.npoints_wbc)
        self.bcvec[0] = self.phiminval
        self.bcvec[-1] = self.delta_x * self.phimaxval

    
    def initializeRhoVector1D(self, rho_function):
        self.rhovec = np.zeros(self.npoints_wbc)
        self.rhovec = -np.power(self.delta_x, 2) * rho_function(self.xgrid[1:-1])
    
    
    def initializeMatrices1D(self):
        self.A = np.zeros([self.npoints_wbc, self.npoints_wbc])
        if (self.bctype == 0): # Dirichlet
            self.A[0,0] = -2.0
            self.A[0,1] = 1.0
            self.A[-1,-1] = -2.0
            self.A[-1,-2] = 1.0
        elif (self.bctype == 1): # Left Neumann Right Dirichlet
            self.A[0,0] = -1.0
            self.A[0,1] = 1.0
            self.A[-1,-1] = -2.0
            self.A[-1,-2] = 1.0
        elif (self.bctype == 2): # Left Dirichlet Right Neumann
            self.A[0,0] = -2.0
            self.A[0,1] = 1.0
            self.A[-1,-1] = -1.0
            self.A[-1,-2] = 1.0

        for i in range(1,self.npoints_wbc-1):
            self.A[i,i-1] = 1.0
            self.A[i,i] = -2.0
            self.A[i,i+1] = 1.0

        self.B = self.rhovec - self.bcvec
    
    
    def setupSystem(self, xmin, xmax, npoints, bctype, phiminval, phimaxval, rho_function, print_times, potmask=None):
        if (potmask==None):
            print("Default")
        self.print_times = print_times

        start_grid = time.time()
        self.initializeGrid(xmin, xmax, npoints)
        end_grid = time.time() - start_grid

        start_bc = time.time()
        self.initializeBoundaryCondition(bctype, phiminval, phimaxval)
        end_bc = time.time() - start_bc

        start_rho = time.time()
        self.initializeRhoVector1D(rho_function)
        end_rho = time.time() - start_rho

        start_matrices = time.time()
        self.initializeMatrices1D()
        end_matrices = time.time() - start_matrices
        end_setup = time.time() - start_grid

        if (print_times == 1):
            print("Time taken for setup: ", end_setup, "s")
            print("Breakdown")
            print("--------------------------------------------")
            print("   Time for setting up grid: ", end_grid, "s")
            print("   Time for setting boundary conditions: ", end_bc, "s")
            print("   Time for sampling charge density: ", end_rho, "s")
            print("   Time for building A and B matrices: ", end_matrices, "s")
            print("--------------------------------------------")
    
    def solve(self, use_gpu, use_sparse):

        start_solve = time.time()
        if (use_gpu == 1 and use_sparse == 0):
            import cupy
            cupy.show_config()
            A_gpu = cupy.asarray(self.A)
            B_gpu = cupy.asarray(self.B)
            transfer_time = time.time() - start_solve
            solve_time = time.time()
            sol_gpu = cupy.linalg.solve(A_gpu, B_gpu)
            end_solve_time = time.time() - solve_time
            transfer_to_cpu = time.time()
            self.phi[1:-1] = cupy.asnumpy(sol_gpu)
            end_transfer = time.time() - transfer_to_cpu
        elif (use_gpu == 1 and use_sparse == 1):
            import cupy
            import cupyx.scipy.sparse.linalg
            import cupyx.scipy.sparse
            #cupy.show_config()
            A_gpu = cupyx.scipy.sparse.csr_matrix(csr_matrix(self.A))
            B_gpu = cupy.asarray(self.B)
            transfer_time = time.time() - start_solve
            solve_time = time.time()
            sol_gpu = cupyx.scipy.sparse.linalg.spsolve(A_gpu, B_gpu)
            end_solve_time = time.time() - solve_time
            transfer_to_cpu = time.time()
            self.phi[1:-1] = cupy.asnumpy(sol_gpu)
            end_transfer = time.time() - transfer_to_cpu
        elif (use_gpu == 0 and use_sparse == 1):
            self.phi[1:-1] = spsolve(csr_matrix(self.A), self.B)
        elif (use_gpu == 0 and use_sparse == 0):
            self.phi[1:-1] = np.linalg.solve(self.A, self.B)
        end_solve = time.time() - start_solve

        if (self.bctype == 0): # Dirichlet
            self.phi[0] = self.phiminval
            self.phi[-1] = self.phimaxval
        elif (self.bctype == 1): # Left Neumann Right Dirichlet
            self.phi[0] = self.phi[1] - self.delta_x*self.phiminval
            self.phi[-1] = self.phimaxval
        elif (self.bctype == 2): # Left Dirichlet Right Neumann
            self.phi[0] = self.phiminval
            self.phi[-1] = self.phi[-2] + self.delta_x*self.phimaxval

        if (self.print_times == 1):
            print("Time to solve: ", end_solve, "s")
            if (use_gpu == 1):
                print("Time to transfer to GPU: ",transfer_time,"s")
                print("Time to actually solve: ", end_solve_time, "s")
                print("Time to transfer back to CPU: ", end_transfer, "s")


    
    def calculateField(self):
        start_field = time.time()
        self.E = np.zeros(self.npoints)
        self.E[0] = (self.phi[1] - self.phi[0])/(self.delta_x)
        for i in range(1,self.npoints-1):
            self.E[i] = (self.phi[i+1] - self.phi[i-1])/(2*self.delta_x)
        self.E[-1] = (self.phi[-1] - self.phi[-2])/(self.delta_x)
        end_field = time.time() - start_field
        if (self.print_times == 1):
            print("Time taken to calculate field: ",end_field,"s")