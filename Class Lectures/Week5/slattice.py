import numpy as np
from numpy.typing import ArrayLike, NDArray
from functools import cached_property
import itertools
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self
import math

class SLattice:
    def __init__(
        self,
        matrix: ArrayLike,
        pbc: tuple[bool, bool, bool] = (True, True, True),
    ) -> None:
        """Create a lattice from any sequence of 9 numbers. Note that the sequence
        is assumed to be read one row at a time. Each row represents one
        lattice vector.

        Args:
            matrix: Sequence of numbers in any form. Examples of acceptable
                input.
                i) An actual numpy array.
                ii) [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                iii) [1, 0, 0 , 0, 1, 0, 0, 0, 1]
                iv) (1, 0, 0, 0, 1, 0, 0, 0, 1)
                Each row should correspond to a lattice vector.
                e.g. [[10, 0, 0], [20, 10, 0], [0, 0, 30]] specifies a lattice
                with lattice vectors [10, 0, 0], [20, 10, 0] and [0, 0, 30].
            pbc: a tuple defining the periodic boundary conditions along the three
                axis of the lattice.
        """
        mat = np.array(matrix, dtype=np.float64).reshape((3, 3))
        mat.setflags(write=False)  # make the matrix immutable
        self._matrix: NDArray[np.float64] = mat
        self._inv_matrix: NDArray[np.float64] | None = None

        self.pbc = pbc

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Lattice",
                f"    abc : {' '.join(map(repr, self.lengths))}",
                f" angles : {' '.join(map(repr, self.angles))}",
                f" volume : {self.volume!r}",
                f"      A : {' '.join(map(repr, self._matrix[0]))}",
                f"      B : {' '.join(map(repr, self._matrix[1]))}",
                f"      C : {' '.join(map(repr, self._matrix[2]))}",
                f"    pbc : {' '.join(map(repr, self.pbc))}",
            ]
        )

    def __eq__(self, other: object) -> bool:
        """A lattice is considered to be equal to another if the internal matrix
        representation satisfies np.allclose(matrix1, matrix2) and
        share the same periodicity.
        """
        if not hasattr(other, "matrix") or not hasattr(other, "pbc"):
            return NotImplemented

        # Shortcut the np.allclose if the memory addresses are the same
        # (very common in Structure.from_sites)
        if self is other:
            return True

        return np.allclose(self.matrix, other.matrix) and self.pbc == other.pbc

    def __hash__(self) -> int:
        return hash((self.lengths, self.angles, self.pbc))

    def __str__(self) -> str:
        return "\n".join(" ".join([f"{i:.6f}" for i in row]) for row in self._matrix)

    def __format__(self, fmt_spec: str = "") -> str:
        """Support format printing.

        Supported fmt_spec (str) are:
        1. "l" for a list format that can be easily copied and pasted, e.g.
           ".3fl" prints something like
           "[[10.000, 0.000, 0.000], [0.000, 10.000, 0.000], [0.000, 0.000, 10.000]]"
        2. "p" for lattice parameters ".1fp" prints something like
           "{10.0, 10.0, 10.0, 90.0, 90.0, 90.0}"
        3. Default will simply print a 3x3 matrix form. E.g.
           10 0 0
           0 10 0
           0 0 10
        """
        matrix = self._matrix.tolist()
        if fmt_spec.endswith("l"):
            fmt = "[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]"
            fmt_spec = fmt_spec[:-1]
        elif fmt_spec.endswith("p"):
            fmt = "{{{}, {}, {}, {}, {}, {}}}"
            fmt_spec = fmt_spec[:-1]
            matrix = (self.lengths, self.angles)
        else:
            fmt = "{} {} {}\n{} {} {}\n{} {} {}"

        return fmt.format(*(format(c, fmt_spec) for row in matrix for c in row))

    @property
    def pbc(self) -> tuple[bool, bool, bool]:
        """Tuple defining the periodicity of the Lattice."""
        return self._pbc  # type:ignore[return-value]

    @pbc.setter
    def pbc(self, pbc: tuple[bool, bool, bool]) -> None:
        if len(pbc) != 3 or any(item not in {True, False} for item in pbc):
            raise ValueError(f"pbc must be a tuple of three True/False values, got {pbc}")
        self._pbc = tuple(bool(item) for item in pbc)

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Copy of matrix representing the Lattice."""
        return self._matrix

    @cached_property
    def lengths(self) -> tuple[float, float, float]:
        """Lattice lengths.

        Returns:
            The lengths (a, b, c) of the lattice.
        """
        return tuple(np.sqrt(np.sum(self._matrix**2, axis=1)).tolist())  # type: ignore[return-value]

    @cached_property
    def angles(self) -> tuple[float, float, float]:
        """Lattice angles.

        Returns:
            The angles (alpha, beta, gamma) of the lattice.
        """
        matrix, lengths = self._matrix, self.lengths
        angles = np.zeros(3)
        for dim in range(3):
            jj = (dim + 1) % 3
            kk = (dim + 2) % 3
            angles[dim] = np.clip(np.dot(matrix[jj], matrix[kk]) / (lengths[jj] * lengths[kk]), -1, 1)
        angles = np.arccos(angles) * 180.0 / np.pi  # type: ignore[assignment]
        return tuple(angles.tolist())  # type: ignore[return-value]

    @cached_property
    def volume(self) -> float:
        """Volume of the unit cell in Angstrom^3."""
        matrix = self._matrix
        return float(abs(np.dot(np.cross(matrix[0], matrix[1]), matrix[2])))

    @property
    def is_orthogonal(self) -> bool:
        """Whether all angles are 90 degrees."""
        return all(abs(a - 90) < 1e-5 for a in self.angles)

    @property
    def is_3d_periodic(self) -> bool:
        """True if the Lattice is periodic in all directions."""
        return all(self.pbc)

    @property
    def inv_matrix(self) -> NDArray[np.float64]:
        """Inverse of lattice matrix."""
        if self._inv_matrix is None:
            self._inv_matrix = np.linalg.inv(self._matrix)  # type: ignore[assignment]
            self._inv_matrix.setflags(write=False)
        return self._inv_matrix  # type: ignore[return-value]

    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        """The metric tensor of the lattice."""
        return np.dot(self._matrix, self._matrix.T)

    def copy(self) -> Self:
        """Make a copy of this lattice."""
        return type(self)(self.matrix.copy(), pbc=self.pbc)

    def get_cartesian_coords(self, fractional_coords: ArrayLike) -> NDArray[np.float64]:
        """Get the Cartesian coordinates given fractional coordinates.

        Args:
            fractional_coords (3x1 array): Fractional coords.

        Returns:
            Cartesian coordinates
        """
        return np.dot(fractional_coords, self._matrix)

    def get_fractional_coords(self, cart_coords: ArrayLike) -> NDArray[np.float64]:
        """Get the fractional coordinates given Cartesian coordinates.

        Args:
            cart_coords (3x1 array): Cartesian coords.

        Returns:
            Fractional coordinates.
        """
        return np.dot(cart_coords, self.inv_matrix)

    @classmethod
    def cubic(cls, a: float, pbc: tuple[bool, bool, bool] = (True, True, True)) -> Self:
        """Convenience constructor for a cubic lattice.

        Args:
            a (float): The *a* lattice parameter of the cubic cell.
            pbc (tuple): a tuple defining the periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.

        Returns:
            Cubic lattice of dimensions (a x a x a).
        """
        return cls([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], pbc)

    @classmethod
    def tetragonal(
        cls,
        a: float,
        c: float,
        pbc: tuple[bool, bool, bool] = (True, True, True),
    ) -> Self:
        """Convenience constructor for a tetragonal lattice.

        Args:
            a (float): *a* lattice parameter of the tetragonal cell.
            c (float): *c* lattice parameter of the tetragonal cell.
            pbc (tuple): The periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.

        Returns:
            Tetragonal lattice of dimensions (a x a x c).
        """
        return cls.from_parameters(a, a, c, 90, 90, 90, pbc=pbc)

    @classmethod
    def orthorhombic(
        cls,
        a: float,
        b: float,
        c: float,
        pbc: tuple[bool, bool, bool] = (True, True, True),
    ) -> Self:
        """Convenience constructor for an orthorhombic lattice.

        Args:
            a (float): *a* lattice parameter of the orthorhombic cell.
            b (float): *b* lattice parameter of the orthorhombic cell.
            c (float): *c* lattice parameter of the orthorhombic cell.
            pbc (tuple): a tuple defining the periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.

        Returns:
            Orthorhombic lattice of dimensions (a x b x c).
        """
        return cls.from_parameters(a, b, c, 90, 90, 90, pbc=pbc)

    @classmethod
    def monoclinic(
        cls,
        a: float,
        b: float,
        c: float,
        beta: float,
        pbc: tuple[bool, bool, bool] = (True, True, True),
    ) -> Self:
        """Convenience constructor for a monoclinic lattice.

        Args:
            a (float): *a* lattice parameter of the monoclinic cell.
            b (float): *b* lattice parameter of the monoclinic cell.
            c (float): *c* lattice parameter of the monoclinic cell.
            beta (float): *beta* angle between lattice vectors b and c in
                degrees.
            pbc (tuple): a tuple defining the periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.

        Returns:
            Monoclinic lattice of dimensions (a x b x c) with non
                right-angle beta between lattice vectors a and c.
        """
        return cls.from_parameters(a, b, c, 90, beta, 90, pbc=pbc)

    @classmethod
    def hexagonal(cls, a: float, c: float, pbc: tuple[bool, bool, bool] = (True, True, True)) -> Self:
        """Convenience constructor for a hexagonal lattice.

        Args:
            a (float): *a* lattice parameter of the hexagonal cell.
            c (float): *c* lattice parameter of the hexagonal cell.
            pbc (tuple): a tuple defining the periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.

        Returns:
            Hexagonal lattice of dimensions (a x a x c).
        """
        return cls.from_parameters(a, a, c, 90, 90, 120, pbc=pbc)

    @classmethod
    def rhombohedral(
        cls,
        a: float,
        alpha: float,
        pbc: tuple[bool, bool, bool] = (True, True, True),
    ) -> Self:
        """Convenience constructor for a rhombohedral lattice.

        Args:
            a (float): *a* lattice parameter of the rhombohedral cell.
            alpha (float): Angle for the rhombohedral lattice in degrees.
            pbc (tuple): a tuple defining the periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.

        Returns:
            Rhombohedral lattice of dimensions (a x a x a).
        """
        return cls.from_parameters(a, a, a, alpha, alpha, alpha, pbc=pbc)

    @classmethod
    def from_parameters(
        cls,
        a: float,
        b: float,
        c: float,
        alpha: float,
        beta: float,
        gamma: float,
        *,
        vesta: bool = False,
        pbc: tuple[bool, bool, bool] = (True, True, True),
    ) -> Self:
        """Create a Lattice using unit cell lengths (in Angstrom) and angles (in degrees).

        Args:
            a (float): *a* lattice parameter.
            b (float): *b* lattice parameter.
            c (float): *c* lattice parameter.
            alpha (float): *alpha* angle in degrees.
            beta (float): *beta* angle in degrees.
            gamma (float): *gamma* angle in degrees.
            vesta (bool): True if you import Cartesian coordinates from VESTA.
            pbc (tuple): a tuple defining the periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.

        Returns:
            Lattice with the specified lattice parameters.
        """
        angles_r = np.radians([alpha, beta, gamma])
        cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
        sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

        if vesta:
            c1 = c * cos_beta
            c2 = (c * (cos_alpha - (cos_beta * cos_gamma))) / sin_gamma

            vector_a = [float(a), 0.0, 0.0]
            vector_b = [b * cos_gamma, b * sin_gamma, 0]
            vector_c = [c1, c2, math.sqrt(c**2 - c1**2 - c2**2)]

        else:
            val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
            val = np.clip(val, -1, 1)  # rounding errors may cause values slightly > 1
            gamma_star = np.arccos(val)

            vector_a = [a * sin_beta, 0.0, a * cos_beta]
            vector_b = [
                -b * sin_alpha * np.cos(gamma_star),
                b * sin_alpha * np.sin(gamma_star),
                b * cos_alpha,
            ]
            vector_c = [0.0, 0.0, float(c)]

        return cls([vector_a, vector_b, vector_c], pbc)

    @property
    def a(self) -> float:
        """*a* lattice parameter."""
        return self.lengths[0]

    @property
    def b(self) -> float:
        """*b* lattice parameter."""
        return self.lengths[1]

    @property
    def c(self) -> float:
        """*c* lattice parameter."""
        return self.lengths[2]

    @property
    def abc(self) -> tuple[float, float, float]:
        """Lengths of the lattice vectors, i.e. (a, b, c)."""
        return self.lengths

    @property
    def alpha(self) -> float:
        """Angle alpha of lattice in degrees."""
        return self.angles[0]

    @property
    def beta(self) -> float:
        """Angle beta of lattice in degrees."""
        return self.angles[1]

    @property
    def gamma(self) -> float:
        """Angle gamma of lattice in degrees."""
        return self.angles[2]

    @property
    def parameters(self) -> tuple[float, float, float, float, float, float]:
        """6-tuple of floats (a, b, c, alpha, beta, gamma)."""
        return (*self.lengths, *self.angles)

    @property
    def params_dict(self) -> dict[str, float]:
        """Dictionary of lattice parameters."""
        return dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), self.parameters, strict=True))

    def dot(
        self,
        coords_a: ArrayLike,
        coords_b: ArrayLike,
        frac_coords: bool = False,
    ) -> NDArray[np.float64]:
        """Compute the scalar product of vector(s).

        Args:
            coords_a: Array-like coordinates.
            coords_b: Array-like coordinates.
            frac_coords (bool): True if the vectors are fractional (as opposed to Cartesian) coordinates.

        Returns:
            one-dimensional `numpy` array.
        """
        coords_a, coords_b = np.reshape(coords_a, (-1, 3)), np.reshape(coords_b, (-1, 3))

        if len(coords_a) != len(coords_b):
            raise ValueError("Coordinates must have same length!")

        for coord in (coords_a, coords_b):
            if np.iscomplexobj(coord):
                raise TypeError(f"Complex array are not supported, got {coord=}")

        if not frac_coords:
            cart_a, cart_b = coords_a, coords_b
        else:
            cart_a = np.reshape([self.get_cartesian_coords(vec) for vec in coords_a], (-1, 3))
            cart_b = np.reshape([self.get_cartesian_coords(vec) for vec in coords_b], (-1, 3))

        return np.array(list(itertools.starmap(np.dot, zip(cart_a, cart_b, strict=True))))

    def norm(self, coords: ArrayLike, frac_coords: bool = True) -> NDArray[np.float64]:
        """Compute the norm of vector(s).

        Args:
            coords:
                Array-like object with the coordinates.
            frac_coords:
                Boolean stating whether the vector corresponds to fractional or
                Cartesian coordinates.

        Returns:
            one-dimensional `numpy` array.
        """
        return np.sqrt(self.dot(coords, coords, frac_coords=frac_coords))

tetragonal = SLattice.tetragonal(3.5, 5.0)
print("Tetragonal lattice:\n",tetragonal)
print("")

if __name__ == "__main__":
    tetragonal = SLattice.tetragonal(3.5, 5.0)
    fcoords = [0.5, 0.25, 0.75]
    print(tetragonal.norm(fcoords))
    print(np.sqrt(np.dot(np.dot(tetragonal.metric_tensor, fcoords), fcoords)))  # should be the same