"""NOISEFLOOR: a governed SRE agent that plans inside a JEPA latent world model.

CPU by default. Unless NOISEFLOOR_ALLOW_GPU=1 is set explicitly (only in the Kaggle GPU track),
CUDA is hidden before any module imports torch, so nothing in this package can touch a GPU.
"""
import os

if os.environ.get("NOISEFLOOR_ALLOW_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

__version__ = "0.1.0a1"
