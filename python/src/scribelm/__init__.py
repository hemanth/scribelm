"""
scribelm — Write, rewrite, and critique technical prose in Hemanth HM's voice.
"""

from .engine import (
    ScribeSession,
    critique_text,
    generate_social,
    rewrite_text,
    scribe,
    write_post,
)

__version__ = "0.1.0"
__all__ = [
    "scribe",
    "ScribeSession",
    "critique_text",
    "rewrite_text",
    "write_post",
    "generate_social",
]
