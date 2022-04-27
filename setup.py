import setuptools

with open("README.md", "r") as r:
    long_description = r.read()

setuptools.setup(
    name = "proyecto_1",
    version = "0.0.0",
    author = "Otreblan",
    description = "SVM",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CS2601-2022-1-Grupo-2/proyecto-1",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', # noqa
    ],
    install_requires=[
        'numpy',
        'pandas',
    ],
    entry_points={
        "console_scripts": [
            "proyecto-1 = proyecto_1.main:main",
        ],
    },
)
