import math
import numpy as np

try:
    from tqdm import trange
    _TQDM_AVAILABLE = True
except ImportError:
    trange = range
    _TQDM_AVAILABLE = False


class Ornstein_Uhlenbeck:
    def __init__(self, alpha=8.0, D=0.25, h=1e-3, N=1000):
        self.alpha = alpha
        self.D = D
        self.h = h
        self.N = N

    # ------------------------------------------------------------------
    # Spectral quantities
    # ------------------------------------------------------------------

    def eig_val(self, l, tau=None):
        """l-th Koopman eigenvalue."""
        if tau is None:
            tau = self.h
        return np.exp(-self.alpha * self.D * (l - 1) * tau)

    def probabilistic_hermite(self, l, x):
        """Probabilists' Hermite polynomials He_l(x)."""
        x = np.asarray(x)

        H0 = np.ones_like(x)

        if l == 0:
            return H0

        H1 = x
        if l == 1:
            return H1

        Hm2 = H0
        Hm1 = H1

        for n in range(2, l + 1):
            Hn = x * Hm1 - (n - 1) * Hm2
            Hm2, Hm1 = Hm1, Hn

        return Hn

    def eig_func_KOP(self, l, x):
        """Koopman eigenfunction."""
        return (
            self.probabilistic_hermite(l-1, np.sqrt(self.alpha) * x) / np.sqrt(math.factorial(l - 1))
            )

    def invariant_measure(self, x):
        return (
            np.sqrt(self.alpha / (2 * np.pi))
            * np.exp(-0.5 * self.alpha * x**2)
        )

    def eig_func_POP(self, l, x):
        """Perron-Frobenius eigenfunction."""
        return self.eig_func_KOP(l, x) * self.invariant_measure(x)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, x0, seed=None,N = None, show_progress=False):
        """
        Simulate OU process.

        Parameters
        ----------
        x0 : array_like
            Shape (M,) or scalar.
        """

        if N is None:
            N = self.N
        else:
            self.N = N

        rng = np.random.default_rng(seed)

        x0 = np.asarray(x0, dtype=float)

        a = np.exp(-self.alpha * self.D * self.h)
        var = (1.0 - np.exp(-2.0 * self.alpha * self.D * self.h)) / self.alpha
        std = np.sqrt(var)

        X = np.empty((self.N + 1,) + x0.shape)
        X[0] = x0

        iterator = trange(
            self.N,
            desc="Simulating",
            disable=not (show_progress and _TQDM_AVAILABLE),
        )

        for k in iterator:
            X[k + 1] = a * X[k] + std * rng.normal(size=x0.shape)

        return X

    def step(self, x0, seed=None):
        """Advance the process N steps and return the final state."""
        X = self.simulate(x0, seed=seed)
        return X[-1]

    def simulate_fast(self, x0, seed=None , N = None, show_progress=False):
        """
        Simulate only the final state (memory efficient).
        """
        if N is None:
            N = self.N
        else:
            self.N = N

        rng = np.random.default_rng(seed)

        x = np.asarray(x0, dtype=float).copy()

        a = np.exp(-self.alpha * self.D * self.h)
        var = (1.0 - np.exp(-2.0 * self.alpha * self.D * self.h)) / self.alpha
        std = np.sqrt(var)

        iterator = trange(
            self.N,
            desc="Simulating fast",
            disable=not (show_progress and _TQDM_AVAILABLE),
        )

        for _ in iterator:
            x = a * x + std * rng.normal(size=x.shape)

        return x


if __name__ == "__main__":

    OU = Ornstein_Uhlenbeck(
        alpha=8,
        D=0.25,
        h=1e-3,
        N=10000,
    )

    # single trajectory
    x0 = np.array([0.5])

    X = OU.simulate(x0, seed=42)
    print(X.shape)

    xf = OU.step(x0)
    print(xf.shape)

    # multiple initial conditions
    x0 = np.linspace(-2, 2, 100)

    X = OU.simulate(x0)
    print(X.shape)  # (N+1,100)

    xf = OU.simulate_fast(x0)
    print(xf.shape)  # (100,)