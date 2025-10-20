from setuptools import setup, find_packages

setup(
    name="kazsim",
    version="0.1.0",
    author="Your Name",
    description="Text simplification library for Kazakh language",
    packages=find_packages(),
    install_requires=[
        "unsloth",
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets",
        "pandas",
        "wandb",
        "trl",
        "pyyaml",
    ],
    python_requires=">=3.8",
)