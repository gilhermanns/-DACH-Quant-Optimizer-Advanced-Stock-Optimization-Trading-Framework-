from setuptools import setup, find_packages

setup(
    name="gilhermanns-QuantPortfolio",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas", "numpy", "yfinance", "backtrader", 
        "scikit-learn", "tensorflow", "matplotlib", 
        "seaborn", "gudhi", "statsmodels", "pyportfolioopt"
    ],
    author="Gil Hermanns",
    description="Professional DACH-focused quantitative trading framework.",
    license="MIT",
    keywords="DAX, XETRA, SMI, algorithmic trading, quantitative finance",
)
