from setuptools import setup, find_packages

setup(
    name="advanced-classification-model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
    ],
    author="Mohamed Traore",
    author_email="mt.db@icloud.com",
    description="An advanced classification model using ensemble of classifiers, self-supervised learning, data augmentation, and adaptive training strategies.",
    url="https://github.com/yourusername/advanced-classification-model",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
)
