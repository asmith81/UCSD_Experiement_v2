from setuptools import setup, find_packages

setup(
    name="lmm_invoice_extraction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "Pillow",
        "numpy",
        "pytest",
        "pyyaml"
    ],
) 