"""Bharat-3B Smart-Core: Setup Configuration."""

from setuptools import setup, find_packages

setup(
    name="bharat-3b-smart-core",
    version="0.1.0",
    description="3B parameter LLM with DEQ/RMT/MoS architecture for Indian languages",
    author="Bharat AI Labs",
    author_email="research@bharatailabs.in",
    url="https://github.com/bharat-ai-labs/bharat-3b",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "jax[tpu]>=0.4.30",
        "flax>=0.8.0",
        "optax>=0.2.0",
        "sentencepiece>=0.2.0",
        "einops>=0.8.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-xdist", "black", "ruff"],
        "serve": ["fastapi", "uvicorn", "pydantic"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
    ],
)
