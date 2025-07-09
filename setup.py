from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="diffusion_pinn",
    version="0.2.26.td1",  # Increment version
    description="Physics-Informed Neural Network for Diffusion Problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Quinn Stein",
    author_email="qs8@hw.ac.uk",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.json", "*.mat", "*.csv", "*.py"]},  # Ensure .py files are included
    python_requires=">=3.8",
    install_requires=required,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="physics-informed neural networks, diffusion, machine learning",
    project_urls={
        "Source": "https://github.com/QuinnSteinRes/diffusion_pinn",
    }
)
