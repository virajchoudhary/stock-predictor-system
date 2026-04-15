from setuptools import setup, find_packages

setup(
    name="quant_reporter",
    version="1.0.0",
    packages=find_packages(include=["quant_reporter", "quant_reporter.*"]),
)
