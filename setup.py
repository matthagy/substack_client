from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="substack_client",
    description="Prototype client for interacting with Substack sites",
    version="0.0.1",
    author="Matt Hagy",
    author_email="matthew.hagy@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matthagy/substack_client",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
