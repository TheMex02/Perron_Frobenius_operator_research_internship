import sys

from matplotlib.pylab import eig
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

class EDMD:
    def __init__(self):
        self.X = None
        self.Y = None
        self.dictionary = None

        self.psi = None
        self.PSI = None
        self.A = None
        self.G = None

        self.K_dict = {
                    "operator": None,
                    "eigenvalue": None,
                    "eigenvector": None,
                    "eigenfunction": None,
                    }
        self.P_dict = {
                    "operator": None,
                    "eigenvalue": None,
                    "eigenvector": None,
                    "eigenfunction": None,
                    }

        self.B = None
        self.dim_Domain = None
        self.Domain = None
        self.point_per_dimension = None
    
    def setup(self,X,Y,dictionary):
        self.X = X
        self.Y = Y
        self.dictionary = dictionary

        print("Setting up psi vector...")
        self.psi = psiVector(self.dictionary)

        print("Setting up PSI matrix...")
        self.PSI = PsiMatrix(self.psi)

        print("Setting up A & G matrix...")
        self.A,self.G = self.compute_A_G(self.X, self.Y, self.psi)

        print("Computing operators...")        
        self.compute_operators()

        print("Computing eigenvalues, eigenvectors and eigenfunctions...")
        self.compute_eigenval_vec_func()

    def compute_operators(self):
        # Compute Koopman operator
        #self.K_dict["operator"] = np.linalg.solve(self.G, self.A.T)
        self.K_dict["operator"] = np.linalg.pinv(self.G) @ self.A.T
        # Compute Perron-Frobenius operator
        self.P_dict["operator"] = np.linalg.solve(self.G, self.K_dict["operator"].T @ self.G)

        print("Difference between direct or indirect computation of Perron Frobenius")
        print(np.linalg.norm(np.linalg.solve(self.G, self.A) - self.P_dict["operator"]))
    
    def compute_eigenval_vec_func(self):
        # Koopman operator K
        eigvals_K, eigvecs_K = scipy.linalg.eig(self.K_dict["operator"])

        #idx = np.argsort(-np.abs(eigvals_K))
        idx = np.argsort(-np.real(eigvals_K))

        self.K_dict["eigenvalue"] = eigvals_K[idx]
        self.K_dict["eigenvector"] = eigvecs_K[:, idx]
        self.K_dict["eigenfunction"] = [
            lambda x, v=v: np.dot(self.psi(x), v)
            for v in self.K_dict["eigenvector"].T
        ]

        # Perron-Frobenius operator P
        eigvals_P, eigvecs_P = scipy.linalg.eig(self.P_dict["operator"])

        #idx = np.argsort(-np.abs(eigvals_P))
        idx = np.argsort(-np.real(eigvals_P))

        self.P_dict["eigenvalue"] = eigvals_P[idx]
        self.P_dict["eigenvector"] = eigvecs_P[:, idx]
        self.P_dict["eigenfunction"] = [
            lambda x, v=v: np.dot(self.psi(x), v)
            for v in self.P_dict["eigenvector"].T
        ]

    def density_to_coeff(self, rho, grid):
        """
        Project density onto basis (any dimension).
        """

        grid = np.asarray(grid)
        rho = np.asarray(rho).ravel()

        Psi = np.array([self.psi(x) for x in grid])  # (N, k)

        coeffs, *_ = np.linalg.lstsq(Psi, rho, rcond=None)

        return coeffs
    
    def coeff_to_density(self, c, grid):
        """
        Reconstruct density from EDMD coefficients (any dimension).
        """

        grid = np.asarray(grid)

        Psi = np.array([self.psi(x) for x in grid])  # (N, k)

        rho = Psi @ c
        rho = np.real(rho)

        # normalize (dimension-independent)
        rho /= np.sum(rho)

        return rho
    
    def apply_P(self, rho, grid, n_steps=1):
        """
        EDMD Perron-Frobenius evolution (dimension-free).
        """

        grid = np.asarray(grid)

        P_power = np.linalg.matrix_power(
            self.P_dict["operator"], n_steps
        )

        c = self.density_to_coeff(rho, grid)

        c_next = P_power @ c

        return self.coeff_to_density(c_next, grid)

    def observable_to_coeff(self, f, grid):
        """
        Project observable onto basis.
        
        f: shape (N,)
        grid: shape (N,)
        """

        Psi = np.array([self.psi(np.array([x])) for x in grid])  # (N,k)

        coeffs, *_ = np.linalg.lstsq(Psi, f, rcond=None)
        return coeffs

    def coeff_to_observable(self, c, grid):
        """
        Reconstruct observable from coefficients.
        """

        Psi = np.array([self.psi(np.array([x])) for x in grid])  # (N,k)

        f = Psi @ c
        return np.real(f)

    def apply_K(self, f, grid, n_steps=1):
        """
        Evolve observable using Koopman EDMD.
        """

        K_power = np.linalg.matrix_power(self.K_dict["operator"], n_steps)

        c = self.observable_to_coeff(f, grid)

        c_next = K_power @ c

        return self.coeff_to_observable(c_next, grid)
    
    def compute_A_G(self,X, Y, psi):
        """
        X, Y: lists of samples (same length m)
        psi: PsiVector instance
        """

        m = len(X)
        k = psi.k

        A = np.zeros((k, k))
        G = np.zeros((k, k))

        for xl, yl in tqdm(zip(X, Y), total=len(X)):

            psi_x = psi(xl).reshape(k, 1)
            psi_y = psi(yl).reshape(k, 1)

            A += psi_y @ psi_x.T
            G += psi_x @ psi_x.T

        A /= m
        G /= m

        print("G condition number:", np.linalg.cond(G))

        return A, G

class psiVector:
    def __init__(self, basis):
        self.basis = basis
        self.functions = basis.items()
        self.k = len(self.functions)

        if not all(callable(f) for f in self.functions):
            raise TypeError("All basis elements must be callable functions")

    def __call__(self, x):
        x = np.atleast_1d(x)
        return np.array([f(x) for f in self.functions])
    
class PsiMatrix:
    def __init__(self, psi_vector):
        self.psi_vector = psi_vector

    def __call__(self, X):
        return np.array([self.psi_vector(x) for x in X])


