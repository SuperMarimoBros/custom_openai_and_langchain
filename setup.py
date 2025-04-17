from setuptools import setup, find_packages

setup(
    name='custom_openai_langchain',
    version='0.1.0',
    author='SuperMarimoBros',
    description='A custom wrapper for OpenAI and Langchain',
    packages=find_packages(where='.', include=['custom_openai', 'custom_openai.*', 'custom_langchain', 'custom_langchain.*']),
    package_dir={
        'custom_openai': 'custom_openai',
        'custom_langchain': 'custom_langchain',
    },
    install_requires=[
        "openai>=1.70.0",
        "pydantic<3.0.0,>=2.0.0",
        "langchain-community>=0.2.19"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)
