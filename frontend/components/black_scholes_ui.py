import runpy
from pathlib import Path


_STANDALONE_RENDERER = Path(__file__).resolve().with_name("black_scholes_standalone.py")


def render_black_scholes_analyzer():
    runpy.run_path(str(_STANDALONE_RENDERER), run_name="__black_scholes_standalone__")
