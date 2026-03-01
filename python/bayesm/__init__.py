"""
bayesm - Bayesian Inference for Marketing/Micro-Econometrics

Python implementation of the bayesm R package.
"""

__version__ = "0.1.0"
__author__ = "Python port based on R package by Peter Rossi et al."

from .constants import BayesmConstants
from .utilities import pandterm, nmat
from .mix_den import mix_den, mix_den_bi
from .mom_mix import mom_mix
from .mnl_hess import mnl_hess
from .e_mix_marg_den import e_mix_marg_den
from .log_marg_den_nr import log_marg_den_nr
from .mnp_prob import mnp_prob
from .llmnp import llmnp
from .llnhlogit import llnhlogit
from .simnhlogit import simnhlogit
from .runireg import runireg
from .runiregGibbs import runiregGibbs
from .rhierLinearModel import rhierLinearModel
from .cluster_mix import cluster_mix
from .plotting import plot_mat, plot_nmix, plot_hcoef
from .summary import summary_mat, summary_var, summary_nmix
from .rmnlIndepMetrop import rmnlIndepMetrop
from .rbprobitGibbs import rbprobitGibbs
from .rordprobitGibbs import rordprobitGibbs
from .rivGibbs import rivGibbs
from .rnegbinRw import rnegbinRw
from .rsurGibbs import rsurGibbs
from .rmvpGibbs import rmvpGibbs
from .rmnpGibbs import rmnpGibbs
from .rnmixGibbs import rnmixGibbs
from .rhierLinearMixture import rhierLinearMixture
from .rhierMnlRwMixture import rhierMnlRwMixture
from .rDPGibbs import rDPGibbs
from .rivDP import rivDP
from .rhierMnlDP import rhierMnlDP
from .rbayesBLP import rbayesBLP
from .rhierNegbinRw import rhierNegbinRw
from .rscaleUsage import rscaleUsage

__all__ = [
    "BayesmConstants",
    "pandterm",
    "nmat",
    "mix_den",
    "mix_den_bi",
    "mom_mix",
    "mnl_hess",
    "e_mix_marg_den",
    "log_marg_den_nr",
    "mnp_prob",
    "llmnp",
    "llnhlogit",
    "simnhlogit",
    "runireg",
    "runiregGibbs",
    "rhierLinearModel",
    "cluster_mix",
    "plot_mat",
    "plot_nmix",
    "plot_hcoef",
    "summary_mat",
    "summary_var",
    "summary_nmix",
    "rmnlIndepMetrop",
    "rbprobitGibbs",
    "rordprobitGibbs",
    "rivGibbs",
    "rnegbinRw",
    "rsurGibbs",
    "rmvpGibbs",
    "rmnpGibbs",
    "rnmixGibbs",
    "rhierLinearMixture",
    "rhierMnlRwMixture",
    "rDPGibbs",
    "rivDP",
    "rhierMnlDP",
    "rbayesBLP",
    "rhierNegbinRw",
    "rscaleUsage",
]
