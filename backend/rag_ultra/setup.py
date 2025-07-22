from setuptools import setup, find_packages

setup(
    name="rag_ultra",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pypdf",
        "python-docx",
        "python-pptx",
        "pillow",
        "litellm",
    ],
) 