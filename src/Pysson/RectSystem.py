import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import time

class TwoDimRectSolver:
    def initializeGrid(self, xmin, xmax, ymin, ymax, nx, ny):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny
        self.nx_wbc = nx - 2
        self.ny_wbc = ny - 2
        self.delta_x = (self.xmax - self.xmin)/(self.nx - 1.0)
        self.delta_y = (self.ymax - self.ymin)/(self.ny - 1.0)
        self.xgrid = np.linspace(self.xmin, self.xmax, self.nx)
        self.ygrid = np.linspace(self.ymin, self.ymax, self.ny)
        self.bigX, self.bigY = np.meshgrid(self.xgrid, self.ygrid)
    
    def initializeBoundaryCondition(self, leftbc, rightbc, bottombc, topbc, phixminf, phixmaxf, phiyminf, phiymaxf):
        self.leftbc = leftbc
        self.rightbc = rightbc
        self.bottombc = bottombc
        self.topbc = topbc
        self.bcvec_xmin = np.zeros(self.ny)
        self.bcvec_xmax = np.zeros(self.ny)
        self.bcvec_ymin = np.zeros(self.nx)
        self.bcvec_ymax = np.zeros(self.nx)
        self.defineBCVectors(phixminf, phixmaxf, phiyminf, phiymaxf)
            
    
    def defineBCVectors(self, phixminf, phixmaxf, phiyminf, phiymaxf):
        # BC Vector at xmin
        for i in range(self.ny):
            self.bcvec_xmin[i] = phixminf(self.ygrid[i])
            self.bcvec_xmax[i] = phixmaxf(self.ygrid[i])

        for i in range(self.nx):
            self.bcvec_ymin[i] = phiyminf(self.xgrid[i])
            self.bcvec_ymax[i] = phiymaxf(self.xgrid[i])
    
    def constructChargeDensityVector(self, rho_density):
        self.rhovec = np.zeros(self.nx_wbc*self.ny_wbc)
        for i in range(1,self.ny-1):
            self.rhovec[(i-1)*self.nx_wbc:i*self.nx_wbc] = -rho_density(self.xgrid[1:-1], self.ygrid[i])
    

    def constructSubMatrices(self):
        self.sub_A_center = np.zeros([self.nx_wbc, self.nx_wbc])
        self.sub_A_bottom_edge = np.zeros([self.nx_wbc, self.nx_wbc])
        self.sub_A_top_edge = np.zeros([self.nx_wbc, self.nx_wbc])
        for i in range(1, self.nx_wbc-1):
            self.sub_A_center[i,i] = -2.0* (1.0/np.power(self.delta_x, 2) +  1.0/np.power(self.delta_y, 2))
            self.sub_A_center[i,i+1] = 1.0/np.power(self.delta_x, 2)
            self.sub_A_center[i,i-1] = 1.0/np.power(self.delta_x, 2)

            self.sub_A_bottom_edge[i,i] = (-2.0/np.power(self.delta_x, 2)) +  (-2.0 + self.bottombc)/np.power(self.delta_y, 2)
            self.sub_A_bottom_edge[i,i+1] = 1.0/np.power(self.delta_x, 2)
            self.sub_A_bottom_edge[i,i-1] = 1.0/np.power(self.delta_x, 2)

            self.sub_A_top_edge[i,i] = (-2.0/np.power(self.delta_x, 2)) +  (-2.0 + self.topbc)/np.power(self.delta_y, 2)
            self.sub_A_top_edge[i,i+1] = 1.0/np.power(self.delta_x, 2)
            self.sub_A_top_edge[i,i-1] = 1.0/np.power(self.delta_x, 2)
        
        # Account for Neumann
        self.sub_A_bottom_edge[0,0] = (-2.0 + self.leftbc)/np.power(self.delta_x, 2) + (-2.0 + self.bottombc)/np.power(self.delta_y, 2)
        self.sub_A_bottom_edge[-1,-1] = (-2.0 + self.rightbc)/np.power(self.delta_x, 2) + (-2.0 + self.bottombc)/np.power(self.delta_y, 2)
        self.sub_A_bottom_edge[0,1] = 1.0/np.power(self.delta_x, 2)
        self.sub_A_bottom_edge[-1,-2] = 1.0/np.power(self.delta_x, 2)

        self.sub_A_top_edge[0,0] = (-2.0 + self.leftbc)/np.power(self.delta_x, 2) + (-2.0 + self.topbc)/np.power(self.delta_y, 2)
        self.sub_A_top_edge[-1,-1] = (-2.0 + self.rightbc)/np.power(self.delta_x, 2) + (-2.0 + self.topbc)/np.power(self.delta_y, 2)
        self.sub_A_top_edge[0,1] = 1.0/np.power(self.delta_x, 2)
        self.sub_A_top_edge[-1,-2] = 1.0/np.power(self.delta_x, 2)

        self.sub_A_center[0,0] = (-2.0 + self.leftbc)/np.power(self.delta_x, 2)  - (2.0/np.power(self.delta_y, 2))
        self.sub_A_center[-1,-1] = (-2.0 + self.rightbc)/np.power(self.delta_x,2) - (2.0/np.power(self.delta_y, 2))
        self.sub_A_center[0,1] = 1.0/np.power(self.delta_x, 2)
        self.sub_A_center[-1,-2] = 1.0/np.power(self.delta_x, 2)

        self.sub_I = np.eye(self.nx_wbc) * (1.0/np.power(self.delta_y, 2))
    
    def constructBVector(self):
        self.bcvec_ymain = np.zeros(self.nx_wbc*self.ny_wbc)
        self.bcvec_xmain = np.zeros(self.nx_wbc*self.ny_wbc)

        if (self.bottombc == 1):
            self.bcvec_ymain[0:self.nx_wbc] = -self.bcvec_ymin[1:-1]/self.delta_y
        else:
            self.bcvec_ymain[0:self.nx_wbc] = self.bcvec_ymin[1:-1]/np.power(self.delta_y, 2)
        
        if (self.topbc == 1):
            self.bcvec_ymain[(self.ny_wbc-1)*self.nx_wbc:self.ny_wbc*self.nx_wbc] = self.bcvec_ymax[1:-1]/self.delta_y
        else:
            self.bcvec_ymain[(self.ny_wbc-1)*self.nx_wbc:self.ny_wbc*self.nx_wbc] = self.bcvec_ymax[1:-1]/np.power(self.delta_y, 2)
        
        if (self.leftbc == 1):
            for i in range(self.ny_wbc):
                self.bcvec_xmain[i*self.nx_wbc] = -self.bcvec_xmin[i+1]/self.delta_x
        else:
            for i in range(self.ny_wbc):
                self.bcvec_xmain[i*self.nx_wbc] = self.bcvec_xmin[i+1]/np.power(self.delta_x, 2)
        
        if (self.rightbc == 1):
            for i in range(self.ny_wbc):
                self.bcvec_xmain[(i+1)*self.nx_wbc-1] =  self.bcvec_xmax[i+1]/self.delta_x
        else:
            for i in range(self.ny_wbc):
                self.bcvec_xmain[(i+1)*self.nx_wbc-1] = self.bcvec_xmax[i+1]/np.power(self.delta_x, 2)
        
        self.B = self.rhovec - self.bcvec_xmain - self.bcvec_ymain

    
    def constructAMatrix(self):
        self.A = np.zeros([self.nx_wbc*self.ny_wbc, self.nx_wbc*self.ny_wbc])
        self.A[0:self.nx_wbc, 0:self.nx_wbc] = self.sub_A_bottom_edge
        self.A[0:self.nx_wbc, self.nx_wbc:2*self.nx_wbc] = self.sub_I
        for i in range(1,self.ny_wbc-1):
            self.A[i*self.nx_wbc:(i+1)*self.nx_wbc, i*self.nx_wbc:(i+1)*self.nx_wbc] = self.sub_A_center
            self.A[i*self.nx_wbc:(i+1)*self.nx_wbc, (i+1)*self.nx_wbc:(i+2)*self.nx_wbc] = self.sub_I
            self.A[i*self.nx_wbc:(i+1)*self.nx_wbc, (i-1)*self.nx_wbc:i*self.nx_wbc] = self.sub_I
        self.A[(self.ny_wbc-1)*self.nx_wbc:self.ny_wbc*self.nx_wbc, 
               (self.ny_wbc-1)*self.nx_wbc:self.ny_wbc*self.nx_wbc] = self.sub_A_top_edge
        self.A[(self.ny_wbc-1)*self.nx_wbc:self.ny_wbc*self.nx_wbc,
               (self.ny_wbc-2)*self.nx_wbc:(self.ny_wbc-1)*self.nx_wbc] = self.sub_I

    def setupSystem(self, xmin, xmax, nx, ymin, ymax, ny, leftbc, rightbc, 
                    bottombc, topbc, phixminf, phixmaxf, phiyminf, phiymaxf, rho_density, print_times):
        self.print_times = print_times

        setup_start = time.time()
        self.initializeGrid(xmin, xmax, ymin, ymax, nx, ny)
        self.initializeBoundaryCondition(leftbc, rightbc, bottombc, topbc, phixminf, phixmaxf, phiyminf, phiymaxf)
        self.constructChargeDensityVector(rho_density)
        bcrho_end = time.time() - setup_start

        matrix_start = time.time()

        self.constructSubMatrices()
        self.constructAMatrix()
        self.constructBVector()
        
        matrix_end = time.time() - matrix_start
        setup_end = time.time() - setup_start

        if (print_times == 1):
            print("Total time for setup: ", setup_end,"s")
            print("Breakdown")
            print("--------------------------------------")
            print("Time for setting up grid and boundary conditions: ", bcrho_end,"s")
            print("Time for building the linear system: ", matrix_end, "s")
            print("--------------------------------------")
    
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
            self.sol = cupy.asnumpy(sol_gpu)
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
            self.sol = cupy.asnumpy(sol_gpu)
            end_transfer = time.time() - transfer_to_cpu
        if (use_gpu == 0 and use_sparse == 1):
            self.sol = spsolve(csr_matrix(self.A), self.B)
        else:
            self.sol = np.linalg.solve(self.A, self.B)
        end_solve = time.time() - start_solve

        start_post = time.time()
        self.extractPotentialFromSolution()
        end_post = time.time() - start_post

        if (self.print_times == 1):
            print("Time taken for solve: ", end_solve,"s")
            print("Time taken for postprocessing: ", end_post,"s")
    
    def extractPotentialFromSolution(self):
        self.phi = np.zeros([self.ny, self.nx])

        for i in range(1,self.ny-1):
            self.phi[i,1:-1] = self.sol[(i-1)*self.nx_wbc:i*self.nx_wbc]
        
        if (self.leftbc == 0):
            self.phi[:,0] = self.bcvec_xmin[:]
        else:
            self.phi[:,0] = self.phi[:,1] - self.delta_x * self.bcvec_xmin[:]
        
        if (self.rightbc == 0):
            self.phi[:,-1] = self.bcvec_xmax[:]
        else:
            self.phi[:,-1] = self.phi[:,-2] + self.delta_x * self.bcvec_xmax[:]
        
        if (self.bottombc == 0):
            self.phi[0,:] = self.bcvec_ymin[:]
        else:
            self.phi[0,:] = self.phi[1,:] - self.delta_y * self.bcvec_ymin[:]
        
        if (self.topbc == 0):
            self.phi[-1,:] = self.bcvec_ymax[:]
        else:
            self.phi[-1,:] = self.phi[-2,:] + self.delta_y * self.bcvec_ymax[:]
    
    def calculateField(self):
        start_field = time.time()
        self.Ex = np.zeros([self.ny, self.nx])
        self.Ey = np.zeros([self.ny, self.nx])

        # Use central differences everyhwere except boundary
        for i in range(self.ny):
            # Left and right boundary
            self.Ex[i,0] = (self.phi[i,1] - self.phi[i,0])/(self.delta_x)
            self.Ex[i,-1] = (self.phi[i,-1] - self.phi[i,-2])/(self.delta_x)
            for j in range(1,self.nx-1):
                self.Ex[i,j] = (self.phi[i,j+1] - self.phi[i,j-1])/(2*self.delta_x)
        
        for i in range(self.nx):
            # Top and Bottom Boundary
            self.Ey[0,i] = (self.phi[1,i] - self.phi[0,i])/(self.delta_y)
            self.Ey[-1,i] = (self.phi[-1,i] - self.phi[-2,i])/(self.delta_y)
            for j in range(1,self.ny-1):
                self.Ey[j,i] = (self.phi[j+1,i] - self.phi[j-1,i])/(2*self.delta_y)
        
        end_field = time.time() - start_field
        print("Time taken for field calculation: ", end_field, "s")