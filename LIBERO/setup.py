# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
readme = path.join(this_directory, "README.md")
if not path.isfile(readme):
    readme = path.join(this_directory, "..", "README.md")
with open(readme, encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="libero",
    packages=[package for package in find_packages() if package.startswith("libero")],
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning",
    author="Anonymous",
    # url="https://github.com/ARISE-Initiative/robosuite",
    author_email="anonymous@example.invalid, anonymous@example.invalid",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
