from setuptools import setup, find_packages

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name="atlaswildfiretool",
    version="1.0",
    description="Atlas Wildfire Tool",
    author="Atlas",
    packages=find_packages(),
    install_requires=requirements,
)
