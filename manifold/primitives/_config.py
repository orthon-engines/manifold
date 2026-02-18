"""Backend toggle — delegates to pmtvs with USE_RUST backward compat."""
import os

USE_RUST = os.environ.get("PMTVS_USE_RUST",
           os.environ.get("USE_RUST", "1")) != "0"
