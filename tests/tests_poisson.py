import unittest
import numpy as np
import matplotlib.pyplot as pl
from Pysson import RectSystem as fdp
import sys as system

class PoissonTest(unittest.TestCase):
    def test_poisson_gaussian_dirichlet(self):
        #charge density
        rho = lambda x,y: 4*(np.exp(-(x**2 + y**2)) - (x**2 + y**2)*np.exp(-(x**2 + y**2)))

        #analytical solution
        solution = lambda x,y: np.exp(-(x**2 + y**2))

        #BCs (Dirichlet)
        phi_xmin = lambda y: np.exp(-(1 + y**2))
        phi_xmax = lambda y: np.exp(-(1 + y**2))
        phi_ymin = lambda x: np.exp(-(1 + x**2))
        phi_ymax = lambda x: np.exp(-(1 + x**2))

        sys = fdp.TwoDimRectSolver()
        sys.setupSystem(xmin=-1, xmax=1, nx=100, ymin=-1, ymax=1, ny=100, leftbc=0, rightbc=0, bottombc=0, topbc=0,
                phixminf = phi_xmin, phixmaxf=phi_xmax, phiyminf = phi_ymin, phiymaxf=phi_ymax, 
                rho_density=rho, print_times=1)
        sys.solve(use_gpu=0, use_sparse=1)


        real_sol = np.zeros([sys.ny, sys.nx])
        for i in range(sys.nx):
            for j in range(sys.ny):
                real_sol[j, i] = solution(sys.xgrid[i], sys.ygrid[j])
        error_mat = np.abs(real_sol - sys.phi)
        print(np.max(error_mat))

        if (len(system.argv) > 1):
            ax = pl.axes(projection='3d')
            ax.plot_surface(sys.bigX, sys.bigY, sys.phi)
            pl.show()
        
        self.assertAlmostEqual(np.max(error_mat), 0, delta=0.001)
    
    def test_poisson_gaussian_leftbottom_neumann(self):
        #charge density
        rho = lambda x,y: 4*(np.exp(-(x**2 + y**2)) - (x**2 + y**2)*np.exp(-(x**2 + y**2)))

        #analytical solution
        solution = lambda x,y: np.exp(-(x**2 + y**2))

        #BCs (Dirichlet/Neumann)
        phi_xmin = lambda y: 2*np.exp(-(1 + y**2))
        phi_xmax = lambda y: np.exp(-(1 + y**2))
        phi_ymin = lambda x: 2*np.exp(-(1 + x**2))
        phi_ymax = lambda x: np.exp(-(1 + x**2))

        sys = fdp.TwoDimRectSolver()
        sys.setupSystem(xmin=-1, xmax=1, nx=200, ymin=-1, ymax=1, ny=200, leftbc=1, rightbc=0, bottombc=1, topbc=0,
                phixminf = phi_xmin, phixmaxf=phi_xmax, phiyminf = phi_ymin, phiymaxf=phi_ymax, 
                rho_density=rho, print_times=1)
        sys.solve(use_gpu=0, use_sparse=1)


        real_sol = np.zeros([sys.ny, sys.nx])
        for i in range(sys.nx):
            for j in range(sys.ny):
                real_sol[j, i] = solution(sys.xgrid[i], sys.ygrid[j])
        error_mat = np.abs(real_sol - sys.phi)
        print(np.max(error_mat))

        if (len(system.argv) > 1):
            ax = pl.axes(projection='3d')
            ax.plot_surface(sys.bigX, sys.bigY, sys.phi)
            pl.show()
        
        self.assertAlmostEqual(np.max(error_mat), 0, delta=0.01)

    def test_poisson_assymetric_topright_assymmetric_neumann(self):
        # solution
        solution = lambda x,y: x**3 * y**3

        # charge density
        rho = lambda x,y: -6*x*y**3 - 6*y*x**3

        # BCs (Dirichlet/Neumann)
        phi_xmin = lambda y: - y**3
        phi_xmax = lambda y: 3* y**3
        phi_ymin = lambda x: - x**3
        phi_ymax = lambda x: 3* x**3

        sys = fdp.TwoDimRectSolver()
        sys.setupSystem(xmin=-1, xmax=1, nx=100, ymin=-1, ymax=1, ny=400, leftbc=0, rightbc=1, bottombc=0, topbc=1,
                phixminf = phi_xmin, phixmaxf=phi_xmax, phiyminf = phi_ymin, phiymaxf=phi_ymax, 
                rho_density=rho, print_times=1)
        sys.solve(use_gpu=0, use_sparse=1)


        real_sol = np.zeros([sys.ny, sys.nx])
        for i in range(sys.nx):
            for j in range(sys.ny):
                real_sol[j, i] = solution(sys.xgrid[i], sys.ygrid[j])
        error_mat = np.abs(real_sol - sys.phi)
        print(np.max(error_mat))

        if (len(system.argv) > 1):
            ax = pl.axes(projection='3d')
            ax.plot_surface(sys.bigX, sys.bigY, sys.phi)
            pl.show()
        
        self.assertAlmostEqual(np.max(error_mat), 0, delta=0.05)
             

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
