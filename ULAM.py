import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class Ulam:
    def __init__(self, Domain, N=10, n_per_box=10):
            
            # --- Validate Domain ---
            Domain = np.asarray(Domain, dtype=float)
            if Domain.ndim != 2 or Domain.shape[1] != 2:
                raise ValueError("Domain must be of shape (dim, 2), e.g. [[x1_min,x1_max], ...]")
            
            self.dim_Domain = Domain.shape[0]
            self.Domain = Domain

            # --- Validate N ---
            if np.isscalar(N):
                N = np.full(self.dim_Domain, int(N))
            else:
                N = np.asarray(N, dtype=int)
                if N.size != self.dim_Domain:
                    raise ValueError("N must be a scalar or have same length as domain dimension")

            if np.any(N <= 0):
                raise ValueError("All entries of N must be positive")

            self.N = N

            # --- Validate n_per_box ---
            if not np.isscalar(n_per_box) or n_per_box <= 0:
                raise ValueError("n_per_box must be a positive scalar")

            self.n_per_box = int(n_per_box)

            # --- Create grid edges (generalized) ---
            self.edges = []
            for d in range(self.dim_Domain):
                xmin, xmax = Domain[d]
                if xmin >= xmax:
                    raise ValueError(f"Invalid domain bounds in dimension {d}")
                
                edges_d = np.linspace(xmin, xmax, self.N[d] + 1)
                self.edges.append(edges_d)

            # Compute box centers
            self.box_centers = [
                0.5 * (edges[:-1] + edges[1:])
                for edges in self.edges
            ]

            # total number of boxes
            self.n_boxes = np.prod(self.N)

            # --- Placeholders ---
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
            
            self.X = None
            self.Y =None

    def setup(self,X=None,Y=None):
        
        self.P = self.build_P_matrix_fast(X=X,Y=Y)
        
        self.compute_eigenfunctions()

    def get_box_index(self, x):
        x = np.asarray(x, dtype=float)

        if x.size != self.dim_Domain:
            raise ValueError("Point dimension does not match domain dimension")

        indices = []

        for d in range(self.dim_Domain):
            # handle right boundary explicitly
            if x[d] == self.edges[d][-1]:
                idx = self.N[d] - 1
            else:
                idx = np.searchsorted(self.edges[d], x[d], side="right") - 1

            if idx < 0 or idx >= self.N[d]:
                return None

            indices.append(idx)

        return tuple(indices)

    def flatten(self, multi_idx):
        multi_idx = np.asarray(multi_idx, dtype=int)

        if multi_idx.size != self.dim_Domain:
            raise ValueError("Index dimension mismatch")

        if np.any(multi_idx < 0) or np.any(multi_idx >= self.N):
            raise ValueError("Index out of bounds")

        k = 0
        for d in range(self.dim_Domain):
            stride = np.prod(self.N[d+1:]) if d+1 < self.dim_Domain else 1
            k += multi_idx[d] * stride

        return int(k)

    def unflatten(self, k):
        if k < 0 or k >= self.n_boxes:
            raise ValueError("Flat index out of bounds")

        multi_idx = []
        remaining = k

        for d in range(self.dim_Domain):
            stride = np.prod(self.N[d+1:]) if d+1 < self.dim_Domain else 1
            idx_d = remaining // stride
            remaining = remaining % stride
            multi_idx.append(int(idx_d))

        return tuple(multi_idx)

    def indicator(self, multi_idx, x):
        idx = self.get_box_index(x)

        if idx is None:
            return 0

        return 1 if tuple(multi_idx) == idx else 0

    def sample_points(self):
        points = []

        # iterate over all boxes in d dimensions
        for multi_idx in np.ndindex(*self.N):
            samples_per_dim = []

            # sample independently in each dimension
            for d in range(self.dim_Domain):
                x0 = self.edges[d][multi_idx[d]]
                x1 = self.edges[d][multi_idx[d] + 1]

                samples_d = np.random.uniform(x0, x1, self.n_per_box)
                samples_per_dim.append(samples_d)

            # shape: (n_per_box, dim)
            pts = np.stack(samples_per_dim, axis=1)
            points.append(pts)

        self.X = np.vstack(points)
        return self.X
    
    def build_P_matrix_fast(self,X=None,Y=None):
        """
        Build Ulam transition matrix using precomputed images Y = T(X).

        Parameters
        ----------
        Y : array-like, shape (n_samples, d)
            Images of self.X under the map T.
        """
        if X is None:
            if self.X is None:
                raise ValueError("self.X must be initialized before calling this function.")
        else:
            self.X = X

        if Y is None:
            if self.Y is None:
                raise ValueError("self.Y must be initialized before calling this function.")
        else:
            self.Y = Y

        if len(Y) != len(self.X):
            raise ValueError("X and Y must have the same length.")

        P = np.zeros((self.n_boxes, self.n_boxes))

        for x, y in tqdm(zip(self.X, Y), total=len(self.X), desc="Building Ulam matrix (fast)"):
            i0 = self.get_box_index(x)
            if i0 is None:
                continue

            i1 = self.get_box_index(y)
            if i1 is None:
                continue

            k0 = self.flatten(i0)
            k1 = self.flatten(i1)

            P[k0, k1] += 1

        # normalize rows
        row_sums = P.sum(axis=1, keepdims=True)
        P = np.divide(P, row_sums, where=row_sums != 0)

        self.P_dict["operator"] = np.transpose(P)
        self.K_dict["operator"] = P
        return P

    def compute_eigenfunctions(self):
        if self.P_dict["operator"] is None:
            raise ValueError("Build the Ulam matrix first.")

        eigvals, eigvecs = np.linalg.eig(self.P_dict["operator"])

        idx = np.argsort(-np.real(eigvals))

        self.P_dict["eigenvalue"] = eigvals[idx]
        self.P_dict["eigenvector"] = eigvecs[:, idx]

        if self.K_dict["operator"] is None:
            raise ValueError("Build the Ulam matrix first.")

        eigvals, eigvecs = np.linalg.eig(self.K_dict["operator"])

        idx = np.argsort(-np.real(eigvals))

        self.K_dict["eigenvalue"] = eigvals[idx]
        self.K_dict["eigenvector"] = eigvecs[:, idx]

        self.K_dict["eigenfunction"] = [
            self._vector_to_L2_function(v)
            for v in self.K_dict["eigenvector"].T
        ]

        self.P_dict["eigenfunction"] = [
            self._vector_to_L2_function(v)
            for v in self.P_dict["eigenvector"].T
        ]
    
    def compute_box_weights(self, density=None):
        """
        L2 weights for correct inner product.
        If density=None → uniform measure.
        """

        dim = self.dim_Domain
        weights = np.ones(self.n_boxes)

        for k in range(self.n_boxes):
            idx = self.unflatten(k)

            vol = 1.0
            center = []

            for d in range(dim):
                x0 = self.edges[d][idx[d]]
                x1 = self.edges[d][idx[d] + 1]
                vol *= (x1 - x0)
                center.append(0.5 * (x0 + x1))

            if density is not None:
                w = density(np.array(center))
                weights[k] = w * vol
            else:
                weights[k] = vol

        self.box_weights = weights

    def _vector_to_L2_function(self, eigenvector):
        v = np.real(eigenvector)
        w = getattr(self, "box_weights", None)

        def f(x):
            x = np.asarray(x)

            # single point
            if x.ndim == 1:
                idx = self.get_box_index(x)
                if idx is None:
                    return 0.0
                k = self.flatten(idx)
                return v[k]

            # batch
            out = np.zeros(len(x))
            for i, xi in enumerate(x):
                idx = self.get_box_index(xi)
                if idx is None:
                    continue
                k = self.flatten(idx)
                out[i] = v[k]
            return out

        return f
    
    def l2_normalize(self, v):

        w = self.box_weights
        return v / np.sqrt(np.sum(np.abs(v)**2 * w))
    
    def apply_K(self, x):
        """
        Apply the discrete Koopman operator.

        Parameters
        ----------
        x : ndarray
            Array with shape self.N (e.g. (40,40)) representing
            values on the grid.

        Returns
        -------
        ndarray
            Transformed array with the same shape as x.
        """
        if self.K_dict["operator"] is None:
            raise ValueError("Koopman operator has not been built yet.")

        x = np.asarray(x, dtype=float)

        if tuple(x.shape) != tuple(self.N):
            raise ValueError(
                f"Input must have shape {tuple(self.N)}, got {x.shape}"
            )

        x_flat = np.ravel(x, order="C")
        y_flat = self.K_dict["operator"] @ x_flat
        return np.reshape(y_flat, self.N, order="C")

    def apply_P(self, x):
        """
        Apply the discrete Perron-Frobenius operator.

        Parameters
        ----------
        x : ndarray
            Array with shape self.N (e.g. (40,40)) representing
            values on the grid.

        Returns
        -------
        ndarray
            Transformed array with the same shape as x.
        """
        if self.P_dict["operator"] is None:
            raise ValueError("Perron-Frobenius operator has not been built yet.")

        x = np.asarray(x, dtype=float)

        if tuple(x.shape) != tuple(self.N):
            raise ValueError(
                f"Input must have shape {tuple(self.N)}, got {x.shape}"
            )

        x_flat = x.ravel()
        y_flat = self.P_dict["operator"] @ x_flat

        return y_flat.reshape(tuple(self.N))