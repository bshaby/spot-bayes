"""
Configuration for the unified spatial-extremes sampler.

This package merges Shi's GPD scale-mixture model (multiplicative Log-Laplace
nugget, nonstationary Matern covariance) with optional features from Wang's
non-convex-domain model (VisGP covariance, visibility-filtered Wendland kernels,
anisotropy). All three model types share Shi's GPD + nugget tail scheme.

`ModelConfig` is the single object that selects the model and toggles the
optional features. It is intentionally a plain dataclass so it can be pickled
alongside MCMC checkpoints and re-loaded by post-processing scripts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# Model-type identifiers ------------------------------------------------------
MODEL_SHI = "shi"            # scale-aware: nonstationary Matern, knot-based R(s), phi(s)
MODEL_WANG = "wang"          # non-convex: VisGP covariance, visibility Wendland
MODEL_HW = "huser-wadsworth" # single global Levy scale R(t), scalar delta
MODEL_TYPES = (MODEL_SHI, MODEL_WANG, MODEL_HW)

# Tail-exponent (phi / delta) domains ----------------------------------------
PHI_UNIT = "unit"           # phi in (0, 1)   -- Shi's original parameterization
PHI_POSITIVE = "positive"   # phi in (0, inf) -- Wang's wider domain
PHI_DOMAINS = (PHI_UNIT, PHI_POSITIVE)
BARRIER_VISIBILITY = ("obstacle", "domain")


@dataclass
class ModelConfig:
    """Selects the model variant and toggles optional features.

    All three ``model_type`` values run under Shi's GPD + multiplicative
    Log-Laplace nugget scheme. The cross-cutting options (``phi_domain``,
    ``use_anisotropy``) are valid regardless of ``model_type`` (with the one
    documented exception that anisotropy needs a knot-based R(s), so it is not
    available for the Huser-Wadsworth model).
    """

    model_type: str = MODEL_SHI

    # --- Cross-cutting options (valid for all model types) ------------------
    # Domain of the tail exponent phi(s) (shi/wang) or scalar delta (HW).
    phi_domain: str = PHI_UNIT
    # Elliptical (anisotropic) Wendland kernels for R(s). shi & wang only.
    use_anisotropy: bool = False

    # --- Wang-model options (ignored unless model_type == "wang") -----------
    use_visgp: bool = False            # VisGP precision for Z(s)
    use_barrier_kernels: bool = False  # visibility-filtered Wendland kernels
    shapefile_path: Optional[str] = None  # domain boundary (water polygon)
    cliques_path: Optional[str] = None    # pre-computed maximal cliques (.rds)
    # What blocks a barrier kernel's site->knot sight line (kernels._visible):
    #   "obstacle" -- the line crosses a barrier (non-water inside the water
    #                 polygon's bounding box). The fix, and the default.
    #   "domain"   -- the line leaves the water polygon at all. The ORIGINAL rule:
    #                 it also blocks every knot placed outside the polygon (Faroe's
    #                 and NZ's outer knot rings got zero weight everywhere). Kept so
    #                 runs fitted under it are post-processed with the kernels they
    #                 were fitted with -- from_dict assigns it to saved configs that
    #                 predate this field.
    barrier_visibility: str = "obstacle"

    # --- Execution options --------------------------------------------------
    use_mpi: bool = True   # parallelize over time replicates with mpi4py
    n_cores: int = 1       # threads for VisGP clique assembly (non-MPI axis)

    def __post_init__(self) -> None:
        self.validate()

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {MODEL_TYPES}, got {self.model_type!r}"
            )
        if self.phi_domain not in PHI_DOMAINS:
            raise ValueError(
                f"phi_domain must be one of {PHI_DOMAINS}, got {self.phi_domain!r}"
            )
        if self.use_anisotropy and self.model_type == MODEL_HW:
            raise ValueError(
                "use_anisotropy is not available for the Huser-Wadsworth model "
                "(it has no knot-based R(s) to make anisotropic)."
            )
        # Wang-only features require the wang model + a domain.
        if self.use_visgp and self.model_type != MODEL_WANG:
            raise ValueError("use_visgp requires model_type='wang'.")
        if self.use_barrier_kernels and self.model_type != MODEL_WANG:
            raise ValueError("use_barrier_kernels requires model_type='wang'.")
        if self.model_type == MODEL_WANG:
            if self.use_visgp and self.cliques_path is None:
                raise ValueError("VisGP covariance requires --cliques (a .rds file).")
            if self.use_barrier_kernels and self.shapefile_path is None:
                raise ValueError("Barrier kernels require --shapefile.")
        if self.barrier_visibility not in BARRIER_VISIBILITY:
            raise ValueError(f"barrier_visibility must be one of {BARRIER_VISIBILITY}, "
                             f"got {self.barrier_visibility!r}")

    # -- convenience flags --------------------------------------------------
    @property
    def phi_is_positive(self) -> bool:
        return self.phi_domain == PHI_POSITIVE

    @property
    def needs_numerical_rw(self) -> bool:
        """Whether a numerical RW-integration fallback is required.

        The cpp header conservatively claims phi < 0.5, but the closed-form
        pRW/qRW/dRW were validated against direct Monte Carlo to within
        sampling error for phi up to 1.5 (GSL analytically continues the upper
        incomplete gamma to negative parameters). The only hard exclusion is
        alpha0 == 1 (already avoided by alpha0_min). So the closed form serves
        both phi domains and no numerical fallback is currently needed
        (validated in tests/test_rw.py).
        """
        return False

    # -- (de)serialization for checkpoints ----------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        # A saved barrier-kernel config without the field was fitted under the
        # original rule; reload it that way so its kernels are reproduced exactly.
        if known.get("use_barrier_kernels") and "barrier_visibility" not in known:
            known["barrier_visibility"] = "domain"
        return cls(**known)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the model-selection arguments to an existing argparse parser.

    Kept separate from the sampler's own arguments so that the simulation and
    scoring scripts can reuse exactly the same flags.
    """
    g = parser.add_argument_group("model selection")
    g.add_argument("--model", dest="model_type", choices=MODEL_TYPES,
                   default=MODEL_SHI, help="Which model variant to fit.")
    g.add_argument("--phi-domain", dest="phi_domain", choices=PHI_DOMAINS,
                   default=PHI_UNIT,
                   help="Domain of tail exponent: 'unit' (0,1) or 'positive' (0,inf).")
    g.add_argument("--use-anisotropy", dest="use_anisotropy", action="store_true",
                   help="Elliptical (anisotropic) Wendland kernels (shi/wang).")
    g.add_argument("--use-visgp", dest="use_visgp", action="store_true",
                   help="VisGP precision matrix for Z(s) (wang only).")
    g.add_argument("--use-barrier-kernels", dest="use_barrier_kernels",
                   action="store_true",
                   help="Visibility-filtered Wendland kernels (wang only).")
    g.add_argument("--shapefile", dest="shapefile_path", type=str, default=None,
                   help="Path to domain boundary shapefile (wang only).")
    g.add_argument("--cliques", dest="cliques_path", type=str, default=None,
                   help="Path to pre-computed cliques .rds (wang VisGP only).")
    return parser


def config_from_args(args: argparse.Namespace) -> ModelConfig:
    """Build a ModelConfig from parsed argparse args (only the config fields)."""
    fields = ModelConfig.__dataclass_fields__
    kwargs = {k: getattr(args, k) for k in fields if hasattr(args, k)}
    return ModelConfig(**kwargs)
