import unittest
import numpy as np
from Pysson import RectSystem as fdp
import matplotlib.pyplot as pl
import sys as system

class LaplaceTest(unittest.TestCase):
    def test_dirichlet_linear_sparse_2d(self):
        # BCs
        phi_xmin = lambda y: y
        phi_xmax = lambda y: y
        phi_ymin = lambda x: 0
        phi_ymax = lambda x: 1
        rho = lambda x,y: 0

        # solution
        solution = lambda x,y: y
        sys = fdp.TwoDimRectSolver()
        sys.setupSystem(xmin=0, xmax=1, nx=100, ymin=0, ymax=1, ny=100, leftbc=0, rightbc=0, bottombc=0, topbc=0, 
                phixminf=phi_xmin, phixmaxf=phi_xmax, phiyminf=phi_ymin, phiymaxf=phi_ymax, rho_density=rho, print_times=1)
        sys.solve(use_gpu=0 ,use_sparse=1)
        
        real_sol = np.zeros([sys.ny, sys.nx])
        for i in range(sys.nx):
            for j in range(sys.ny):
                real_sol[j, i] = solution(sys.xgrid[i], sys.ygrid[j])
        error_mat = np.abs(real_sol - sys.phi)
        print(np.max(error_mat))
        self.assertAlmostEqual(np.max(error_mat), 0, delta=0.001)
        
        if (len(system.argv) > 1):
            ax = pl.axes(projection='3d')
            ax.plot_surface(sys.bigX, sys.bigY, sys.phi)
            pl.show()
    
    def test_dirichlet_nonlinear_sparse_2d(self):
        # BCs
        phi_xmin = lambda y: y
        phi_xmax = lambda y: 1
        phi_ymin = lambda x: x
        phi_ymax = lambda x: 1
        rho = lambda x,y: 0

        # solution
        solution = lambda x,y: x - x*y + y
        sys = fdp.TwoDimRectSolver()
        sys.setupSystem(xmin=0, xmax=1, nx=100, ymin=0, ymax=1, ny=100, leftbc=0, rightbc=0, bottombc=0, topbc=0, 
                phixminf=phi_xmin, phixmaxf=phi_xmax, phiyminf=phi_ymin, phiymaxf=phi_ymax, rho_density=rho, print_times=1)
        sys.solve(use_gpu=0 ,use_sparse=1)
        
        real_sol = np.zeros([sys.ny, sys.nx])
        for i in range(sys.nx):
            for j in range(sys.ny):
                real_sol[j, i] = solution(sys.xgrid[i], sys.ygrid[j])
        error_mat = np.abs(real_sol - sys.phi)
        print(np.max(error_mat))
        self.assertAlmostEqual(np.max(error_mat), 0, delta=0.001)

        if (len(system.argv) > 1):
            ax = pl.axes(projection='3d')
            ax.plot_surface(sys.bigX, sys.bigY, sys.phi)
            pl.show()
    
    def test_dirichlet_nonlinear_nonuniform_sparse_2d(self):
        # BCs
        phi_xmin = lambda y: y
        phi_xmax = lambda y: 1
        phi_ymin = lambda x: x
        phi_ymax = lambda x: 1
        rho = lambda x,y: 0

        # solution
        solution = lambda x,y: x - x*y + y
        sys = fdp.TwoDimRectSolver()
        sys.setupSystem(xmin=0, xmax=1, nx=47, ymin=0, ymax=1, ny=95, leftbc=0, rightbc=0, bottombc=0, topbc=0, 
                phixminf=phi_xmin, phixmaxf=phi_xmax, phiyminf=phi_ymin, phiymaxf=phi_ymax, rho_density=rho, print_times=1)
        sys.solve(use_gpu=0 ,use_sparse=1)
        
        real_sol = np.zeros([sys.ny, sys.nx])
        for i in range(sys.nx):
            for j in range(sys.ny):
                real_sol[j, i] = solution(sys.xgrid[i], sys.ygrid[j])
        error_mat = np.abs(real_sol - sys.phi)
        print(np.max(error_mat))
        self.assertAlmostEqual(np.max(error_mat), 0, delta=0.001)

        if (len(system.argv) > 1):
            ax = pl.axes(projection='3d')
            ax.plot_surface(sys.bigX, sys.bigY, sys.phi)
            pl.show()
    
    def test_neumann_linear_nonuniform_sparse_2d(self):
        # BCs
        phi_xmin = lambda y: y
        phi_xmax = lambda y: 1
        phi_ymin = lambda x: x
        phi_ymax = lambda x: 1
        rho = lambda x,y: 0

        # solution
        solution = lambda x,y: x + y
        sys = fdp.TwoDimRectSolver()
        sys.setupSystem(xmin=0, xmax=1, nx=47, ymin=0, ymax=1, ny=95, leftbc=0, rightbc=1, bottombc=0, topbc=1,
                phixminf=phi_xmin, phixmaxf=phi_xmax, phiyminf=phi_ymin, phiymaxf=phi_ymax, rho_density=rho, print_times=1)
        sys.solve(use_gpu=0, use_sparse=1)

        real_sol = np.zeros([sys.ny, sys.nx])
        for i in range(sys.nx):
            for j in range(sys.ny):
                real_sol[j, i] = solution(sys.xgrid[i], sys.ygrid[j])
        error_mat = np.abs(real_sol - sys.phi)
        print(np.max(error_mat))
        ax = pl.axes(projection='3d')
        self.assertAlmostEqual(np.max(error_mat), 0, delta=0.001)

        if (len(system.argv) > 1):
            ax.plot_surface(sys.bigX, sys.bigY, sys.phi)
            pl.show()
        

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
        
