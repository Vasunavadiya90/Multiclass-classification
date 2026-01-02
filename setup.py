import os,sys
from setuptools import find_packages,setup
from typing import List
## it helps to find requirements and packages 

def get_requirements(file_path:str)-> List :
    HYphen_e = '-e .'

    requirements= []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        # Filter out comments, empty lines, and -e .
        if HYphen_e in requirements:
            requirements.remove(HYphen_e)
    return requirements

setup(

    name = 'Classification_project',
    version='0.0.1',
    author='Vasu',
    author_email = 'Vasunavadiya933@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')

)