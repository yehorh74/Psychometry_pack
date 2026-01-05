from setuptools import setup, find_packages

setup(
    name="psychometry_pack",             # nazwa paczki (unikalna w PyPI)
    version="0.1.0",
    description="Pakiet do analizy psychometrycznej",
    author="Yehor Horin",
    author_email="danedred15@gmail.com",
    packages=find_packages(),     # automatycznie znajdzie katalog mypackage i moduły
    install_requires=[            # wymagane pakiety z PyPI
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "factor_analyzer"
    ],
    python_requires='>=3.7',      # wersja Pythona
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)