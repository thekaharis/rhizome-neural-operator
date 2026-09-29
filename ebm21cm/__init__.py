"""Energy-parameterized diffusion for 21cmFAST reionization fields.

A scalar energy E(x | delta, z, theta, sigma) over (x_HI, T_b) slices is
learned by denoising score matching on its gradient; samples are drawn with
an EDM Heun sampler, optionally corrected by energy-based MALA steps.
"""

__version__ = "0.1.0"
