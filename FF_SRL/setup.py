import setuptools

setuptools.setup(
    name="FF-SRL",
    version="0.0.1",
    author="Michał Naskręt",
    author_email="mnaskret@gmail.com",
    description="A Python framework for high-performance differentiable medical simulation",
    url="https://github.com/SanoScience/FF-SRL",
    project_urls={
    },
    long_description="",
    long_description_content_type="text/markdown",
    license="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "warp-lang==0.10.1", "usd-core", "trimesh", "climage", "torch", "pynput", "torchwindow", "matplotlib"],
    python_requires=">=3.7"
)
