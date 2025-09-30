from setuptools import setup , find_packages
from typing import List

def function(path : str)-> List[str]:
    requirement=[]
    with open(path) as file:
        requirement=file.readlines()
        requirement=[i.replace('\n' ,'') for i in requirement ]
        
        if '-e.' in requirement:
            requirement.remove('-e.')
    return requirement
        

setup(
    name='RAG CHATBOT FOR LEGAL CONSULTANCY ',
    version='0.0.1',
    author='Nirabhay Singh Rathod',
    author_email='nirbhay105633016@gmail.com',
    packages=find_packages(),
    requires=function('requirements.txt')
)