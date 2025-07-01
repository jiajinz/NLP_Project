from setuptools import setup, find_packages

setup(
    name="emolex",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="EMOLEX: Emotion and Language Exploration for Mental Health",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jiajinz/NLP_Project/",  # Optional
    project_urls={
        "Bug Tracker": "https://github.com/jiajinz/NLP_Project/issues",
    },
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "tensorflow>=2.11",
        "transformers>=4.35",
        "scikit-learn>=1.2",
        "pandas>=1.5",
        "numpy>=1.24",
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "jupyterlab>=3.5",
        "scipy>=1.10",
        "datasets>=2.14",
        "nltk>=3.8",
        "tqdm>=4.65",
        "pyyaml>=6.0"
    ],
    python_requires=">=3.10,<3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)