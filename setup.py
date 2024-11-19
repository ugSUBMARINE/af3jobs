from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="af3jobs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    description="Module for defining molecular components, modifications, and job configurations for AlphaFold 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Karl Gruber",
    author_email="karl.gruber@uni-graz.at",
    url="https://github.com/ugSUBMARINE/af3jobs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    keywords=["AlphaFold", "molecular biology", "protein modeling", "bioinformatics"],
    license="MIT",
)
