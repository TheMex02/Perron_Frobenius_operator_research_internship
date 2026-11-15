import os
import numpy as np
import matplotlib.pyplot as plt

class Dictionary:
    def __init__(self):
        self._items = []
        self._readable = []
        self.attributes = None

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def items(self):
        return self._items

    def readable(self):
        return self._readable

    def get_readable(self, idx):
        return self._readable[idx]

    def save_plot(self, xlim=(-1, 1), num_points=500, filename=None):
        if self.dim != 1:
            raise ValueError("Visualization is only implemented for dim=1.")

        if filename is None:
            filename = f"{self.__class__.__name__}.png"

        x = np.linspace(*xlim, num_points)

        plt.figure(figsize=(8, 6))

        for f, label in zip(self._items, self._readable):
            y = np.array([f([xi]) for xi in x])
            plt.plot(x, y, label=label)

        plt.title(self.__class__.__name__)
        plt.xlabel("x")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Saved to {os.path.abspath(filename)}")
    
class Monomials(Dictionary):
    def __init__(self, degree, TotalDegree=False, dim=None):
        super().__init__()

        self.TotalDegree = TotalDegree

        if TotalDegree:
            if dim is None:
                raise ValueError("dim must be provided when TotalDegree=True")

            if not isinstance(degree, int):
                raise TypeError("For TotalDegree=True, degree must be a scalar int")

            self.degree = degree
            self.dim = dim
            self.attributes = {"TotalDegree": TotalDegree, "degree": degree, "dim": dim}

        else:
            if isinstance(degree, int):
                degree = [degree]

            if not isinstance(degree, (list, tuple)):
                raise TypeError("Input must be int or list/tuple")

            self.degree = degree
            self.dim = len(degree)
            self.attributes = {"degree": degree, "dim": dim}
        self._build()

    def _build(self):

        seen = set()

        # -------------------------
        # TOTAL DEGREE CASE
        # -------------------------
        if self.TotalDegree:

            def build(idx, exps, remaining):
                if idx == self.dim:

                    exps = tuple(exps)

                    if exps in seen:
                        return
                    seen.add(exps)

                    def monomial(x, exps=exps):
                        result = 1
                        for xi, ei in zip(x, exps):
                            result *= xi ** ei
                        return result

                    self._items.append(monomial)

                    if all(e == 0 for e in exps):
                        self._readable.append("1")
                    else:
                        term = []
                        for i, e in enumerate(exps):
                            if e == 0:
                                continue
                            var = f"x{i+1}"
                            term.append(var if e == 1 else f"{var}^{e}")
                        self._readable.append("".join(term))

                    return

                for e in range(remaining + 1):   
                    build(idx + 1, exps + [e], remaining - e)

            build(0, [], self.degree)
        else:

            def build(idx, exps):
                if idx == len(self.degree):

                    exps = tuple(exps)

                    def monomial(x, exps=exps):
                        result = 1
                        for xi, ei in zip(x, exps):
                            result *= xi ** ei
                        return result

                    self._items.append(monomial)

                    if all(e == 0 for e in exps):
                        self._readable.append("1")
                    else:
                        term = []
                        for i, e in enumerate(exps):
                            if e == 0:
                                continue
                            var = f"x{i+1}"
                            term.append(var if e == 1 else f"{var}^{e}")
                        self._readable.append("".join(term))

                    return

                for e in range(self.degree[idx] + 1):
                    build(idx + 1, exps + [e])

            build(0, [])

class probabilistic_Hermite_polynomials(Dictionary):
    def __init__(self, degree, dim):
        super().__init__()
        self.degree = degree
        self.dim = dim
        self.attributes = {"degree": degree, "dim": dim}
        self._build()

    def _build(self):
        from itertools import product

        for exps in product(range(self.degree + 1), repeat=self.dim):
            def polynomial(x, exps=exps):
                result = 1
                for xi, ei in zip(x, exps):
                    result *= self.probabilistic_Hermite_polynomial(xi, ei)
                return result

            self._items.append(polynomial)

            if all(e == 0 for e in exps):
                self._readable.append("1")
            else:
                term = []
                for i, e in enumerate(exps):
                    if e == 0:
                        continue
                    var = f"H_{e}(x{i+1})"
                    term.append(var)
                self._readable.append(" * ".join(term))
    def probabilistic_Hermite_polynomial(self, xi, ei):
        if ei == 0:
            return 1
        elif ei == 1:
            return xi
        else:
            H_prev = 1
            H_curr = xi
            for n in range(2, ei + 1):
                H_next = xi * H_curr - (n - 1) * H_prev
                H_prev, H_curr = H_curr, H_next
            return H_curr      

class Legendre_polynomials(Dictionary):
    def __init__(self, degree, dim):
        super().__init__()
        self.degree = degree
        self.dim = dim
        self.attributes = {"degree": degree, "dim": dim}
        self._build()

    def _build(self):
        from itertools import product

        for exps in product(range(self.degree + 1), repeat=self.dim):
            def polynomial(x, exps=exps):
                result = 1
                for xi, ei in zip(x, exps):
                    result *= self.legendre_polynomial(xi, ei)
                return result

            self._items.append(polynomial)

            if all(e == 0 for e in exps):
                self._readable.append("1")
            else:
                term = []
                for i, e in enumerate(exps):
                    if e == 0:
                        continue
                    var = f"P_{e}(x{i+1})"
                    term.append(var)
                self._readable.append(" * ".join(term))    
    def legendre_polynomial(self,xi, ei):
        if ei == 0:
            return 1
        elif ei == 1:
            return xi
        else:
            P_prev = 1
            P_curr = xi
            for n in range(2, ei + 1):
                P_next = ((2 * n - 1) * xi * P_curr - (n - 1) * P_prev) / n
                P_prev, P_curr = P_curr, P_next
            return P_curr

class Fourier_functions(Dictionary):
    def __init__(self, degree, dim):
        super().__init__()
        self.degree = degree
        self.dim = dim
        self.attributes = {"degree": degree, "dim": dim}
        self._build()

    def _build(self):
        from itertools import product

        # For every dimension, choose:
        # 0  -> constant 1
        # 1  -> sin(x)
        # 2  -> cos(x)
        # 3  -> sin(2x)
        # 4  -> cos(2x)
        # etc.
        for exps in product(range(self.degree + 1), repeat=self.dim):

            def function(x, exps=exps):
                result = 1.0

                for xi, ei in zip(x, exps):
                    result *= self.fourier_function(xi, ei)

                return result

            self._items.append(function)

            if all(e == 0 for e in exps):
                self._readable.append("1")
            else:
                term = []

                for i, e in enumerate(exps):
                    if e == 0:
                        continue

                    var = self._readable_term(i, e)
                    term.append(var)

                self._readable.append(" * ".join(term))

    def fourier_function(self, xi, ei):
        if ei == 0:
            return 1.0

        # Odd indices -> sin
        # Even indices -> cos
        frequency = (ei + 1) // 2

        if ei % 2 == 1:
            return np.sin(frequency * xi)
        else:
            return np.cos(frequency * xi)

    def _readable_term(self, i, e):
        frequency = (e + 1) // 2
        var = f"x{i + 1}"

        if e % 2 == 1:
            return f"sin({frequency}{var})"
        else:
            return f"cos({frequency}{var})"
        
if __name__ == "__main__":
    
    M = Monomials([2, 1])

    print(len(M))              # number of monomials
    print(M.get_readable(3))  # human-readable form
    print(M.readable())       # all readable forms

    M = Monomials(degree=10, TotalDegree=True, dim=2)

    print(len(M))             # number of monomials
    print(M.get_readable(3))  # human-readable form
    print(M.readable())       # all readable forms
    print(M.attributes)       # attributes of the dictionary

    M = Monomials(degree=5, TotalDegree=True, dim=1)

    print(len(M))             # number of monomials
    print(M.get_readable(3))  # human-readable form
    print(M.readable())       # all readable forms
    print(M.attributes)       # attributes of the dictionary

    M.save_plot(filename="monomials_plot.png")  # Save the plot of monomials

    H = probabilistic_Hermite_polynomials(degree=5, dim=1)
    
    print(len(H))             # number of monomials
    print(H.get_readable(3))  # human-readable form
    print(H.readable())       # all readable forms
    print(H.attributes)       # attributes of the dictionary

    H.save_plot(xlim=(-3, 3),filename="hermite_polynomials_plot.png")  # Save the plot of Hermite polynomials

    L = Legendre_polynomials(degree=5, dim=1)
    print(len(L))             # number of monomials
    print(L.get_readable(3))  # human-readable form
    print(L.readable())       # all readable forms
    print(L.attributes)       # attributes of the dictionary

    L.save_plot(filename="legendre_polynomials_plot.png")  # Save the plot of Legendre polynomials