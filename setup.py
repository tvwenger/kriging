from setuptools import setup

setup(
    name="kriging",
    version="1.0",
    description="Ordinary and universal kriging in N dimensions",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=["kriging"],
    install_requires=["numpy", "matplotlib", "scipy"],
)
