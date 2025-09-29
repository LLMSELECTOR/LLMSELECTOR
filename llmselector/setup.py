# setup.py

from setuptools import setup, find_packages

setup(
    name="llmselector",  # Name of your package
    version="0.0.1",           # Version of the package
    packages=find_packages(),  # Automatically find packages
    install_requires=[         # List your dependencies here
        "numpy==1.26.4",              
        "pandas==2.2.2",
        "sqlitedict==2.1.0",
        "tqdm==4.67.1",
        "google-generativeai==0.8.5",
        "openai==1.76.2",
        "anthropic==0.68.0",
        "together==1.2.11",
        "pyarrow==16.1.0",
        "scikit-learn==1.5.1",
        "plotly==5.18.0",
        "jupyterlab-widgets==3.0.11",
        "kaleido==0.2.1",
        "notebook==7.0.8",
        "notebook-shim==0.2.3",
        "pillow==10.4.0",
        "jupyterlab==4.0.11",
        "datasets==3.2.0",
        #"jupyterlab-plotly==5.22.0",  # Add this for JupyterLab compatibility

    ],
    author="Lingjiao Chen",
    author_email="lingjiao@stanford.edu",
    description="llmselector is a framework that optimizes model selection for compound AI systems",
    long_description=open('README.md').read(),  # Optional, if you have a README
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your_package",  # Optional, if applicable
)
