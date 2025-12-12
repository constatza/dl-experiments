"""Pydantic configuration dataclasses for data generation strategies.

Each strategy has a corresponding config dataclass that defines:
- Required and optional parameters
- Type annotations for all parameters
- Default values
- Validation rules (via Pydantic)

All configs use `frozen=True` for immutability and `extra="forbid"` to reject
unknown parameters at validation time.
"""

from typing import Literal

import numpy as np
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config={"extra": "forbid", "arbitrary_types_allowed": True})
class ResidualErrorConfig:
    """Configuration for CG residual error strategy.

    Generates warm-start vectors by running a few CG iterations and capturing
    the final residual error after a fixed number of iterations.

    Args:
        samples: Number of samples to generate.
        residual_iters: Number of CG iterations to run.
        seed: Random seed for reproducibility.
        archive_solutions: Optional pre-computed solution vectors.
        archive_rhs: Optional pre-computed RHS vectors.
    """

    samples: int
    residual_iters: int = 8
    seed: int = 42
    archive_solutions: np.ndarray | None = None
    archive_rhs: np.ndarray | None = None


@dataclass(frozen=True, config={"extra": "forbid", "arbitrary_types_allowed": True})
class ResidualTraceConfig:
    """Configuration for CG residual trace strategy.

    Generates warm-start vectors by running CG iterations and capturing
    intermediate residual vectors along the convergence path.

    Args:
        samples: Number of samples to generate.
        residual_iters: Number of CG iterations to run.
        seed: Random seed for reproducibility.
        archive_solutions: Optional pre-computed solution vectors.
        archive_rhs: Optional pre-computed RHS vectors.
    """

    samples: int
    residual_iters: int = 8
    seed: int = 42
    archive_solutions: np.ndarray | None = None
    archive_rhs: np.ndarray | None = None


@dataclass(frozen=True, config={"extra": "forbid"})
class KrylovConfig:
    """Configuration for Krylov subspace strategy.

    Generates warm-start vectors from the Krylov subspace by running
    a specified number of iterations and extracting basis vectors.

    Args:
        samples: Number of samples to generate.
        krylov_iters: Dimension of Krylov subspace (number of iterations).
        seed: Random seed for reproducibility.
    """

    samples: int
    krylov_iters: int = 15
    seed: int = 42


@dataclass(frozen=True, config={"extra": "forbid"})
class EigenvectorForwardConfig:
    """Configuration for forward eigenvector strategy.

    Generates warm-start vectors from linear combinations of eigenvectors
    corresponding to eigenvalues (smallest, largest, or both).

    Args:
        samples: Number of samples to generate. Defaults to matrix dimension.
        which: Which eigenvectors to use ("smallest", "largest", or "both").
        include_eigenvectors: Whether to include pure eigenvectors in output.
        num_eigenvectors: Number of eigenvectors to use. None or -1 means all.
        seed: Random seed for reproducibility.
    """

    samples: int
    which: Literal["smallest", "largest", "both"] = "smallest"
    include_eigenvectors: bool = False
    num_eigenvectors: int | None = None
    seed: int = 42


@dataclass(frozen=True, config={"extra": "forbid"})
class EigenvectorInverseConfig:
    """Configuration for inverse eigenvector strategy.

    Generates warm-start vectors from linear combinations of eigenvectors
    with inverse eigenvalue weighting.

    Args:
        samples: Number of samples to generate. Defaults to matrix dimension.
        which: Which eigenvectors to use ("smallest", "largest", or "both").
        include_eigenvectors: Whether to include pure eigenvectors in output.
        num_eigenvectors: Number of eigenvectors to use. None or -1 means all.
        seed: Random seed for reproducibility.
    """

    samples: int
    which: Literal["smallest", "largest", "both"] = "smallest"
    include_eigenvectors: bool = False
    num_eigenvectors: int | None = None
    seed: int = 42


@dataclass(frozen=True, config={"extra": "forbid"})
class RandomNormalConfig:
    """Configuration for random normal strategy.

    Generates warm-start vectors by sampling from a normal distribution
    with configurable scale.

    Args:
        samples: Number of samples to generate.
        target_rhs_scale: Scale factor for normal distribution sampling.
        seed: Random seed for reproducibility.
    """

    samples: int
    target_rhs_scale: float = 1.0
    seed: int = 42


@dataclass(frozen=True, config={"extra": "forbid"})
class RhsArchiveConfig:
    """Configuration for RHS archive strategy.

    Loads RHS vectors from files matching a glob pattern, optionally
    solves the linear systems, and returns the solutions.

    Args:
        rhs_glob: Glob pattern for RHS files (required).
        samples: Number of files to load. -1 means all available files.
        shuffle: Whether to shuffle file selection.
        seed: Random seed for shuffling. Required if shuffle=True.
        solve_systems: Whether to solve the linear systems.
        cg_tolerance: CG relative tolerance for solving.
        cg_max_iters: CG maximum iterations for solving.
    """

    rhs_glob: str
    samples: int = -1
    shuffle: bool = False
    seed: int | None = None
    solve_systems: bool = True
    cg_tolerance: float = 1e-12
    cg_max_iters: int = 500


@dataclass(frozen=True, config={"extra": "forbid"})
class SolutionArchiveConfig:
    """Configuration for solution archive strategy.

    Loads solution vectors from files matching a glob pattern.

    Args:
        solutions_glob: Glob pattern for solution files (required).
        samples: Number of files to load. -1 means all available files.
        shuffle: Whether to shuffle file selection.
        seed: Random seed for shuffling. Required if shuffle=True.
    """

    solutions_glob: str
    samples: int = -1
    shuffle: bool = False
    seed: int | None = None
