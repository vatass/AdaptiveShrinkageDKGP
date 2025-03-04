from setuptools import setup, find_packages

setup(
    name="adaptive_shrinkage_dkgp",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "xgboost>=1.5.0",
        "gpytorch>=1.6.0",
        "pandas>=0.25.0,<1.2.0",
        "matplotlib>=3.0.0,<3.3.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Kernel Gaussian Process with Adaptive Shrinkage Estimation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AdaptiveShrinkageDKGP",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 