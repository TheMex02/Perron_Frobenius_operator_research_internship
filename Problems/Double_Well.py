import numpy as np

try:
    from tqdm import trange
    _TQDM_AVAILABLE = True
except ImportError:
    trange = range
    _TQDM_AVAILABLE = False

class Double_Well:
    def __init__(self, sigma=0.7, h=1e-3, N=10000):
        self.sigma = sigma
        self.h = h
        self.N = N

    # potential (optional utility)
    def V(self, X):
        x = X[:, 0]
        y = X[:, 1]
        return (x**2 - 1)**2 + y**2

    # drift = -grad V
    def mu(self, X):
        x = X[:, 0]
        y = X[:, 1]

        return np.stack([
            -4 * x * (x**2 - 1),
            -2 * y
        ], axis=1)

    # main simulation
    def simulate(self, x0, seed=None, show_progress=False):
        rng = np.random.default_rng(seed)

        x0 = np.asarray(x0)
        M = x0.shape[0]

        X = np.empty((self.N + 1, M, 2))
        X[0] = x0

        dW = rng.normal(
            0.0,
            np.sqrt(self.h),
            size=(self.N, M, 2)
        )

        iterator = trange(self.N, desc='Simulating', disable=not (show_progress and _TQDM_AVAILABLE))
        for k in iterator:
            X[k+1] = (
                X[k]
                + self.h * self.mu(X[k])
                + self.sigma * dW[k]
            )

        return X
    
    def step(self, x0, seed=None):
        x0 = np.asarray(x0)

        if x0.ndim == 1:
            x0 = x0[None, :]   # (1,2)

        X = self.simulate(x0, seed=seed)

        return np.asarray(X[-1]).reshape(x0.shape[0], 2)
    
    def simulate_fast(self, x0, seed=None, show_progress=False):
        rng = np.random.default_rng(seed)

        x_curr = np.asarray(x0).copy()
        M = x_curr.shape[0]

        iterator = trange(self.N, desc='Simulating fast', disable=not (show_progress and _TQDM_AVAILABLE))
        for _ in iterator:
            dW = rng.normal(
                0.0,
                np.sqrt(self.h),
                size=(M, 2)
            )

            x_next = (
                x_curr
                + self.h * self.mu(x_curr)
                + self.sigma * dW
            )

            x_curr = x_next  # move forward

        return x_curr
        
if __name__ == "__main__":
    DW = Double_Well(sigma=0.7, h=1e-3, N=10000)

    M = 1
    x0 = np.random.uniform(-2, 2, size=(M, 2))

    X = DW.simulate(x0, seed=42)

    print(X.shape)

    X = DW.step(x0)

    print(X.shape)
