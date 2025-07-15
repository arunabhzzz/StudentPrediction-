from setuptools import setup,find_packages
HYPHEN_E_DOT='-e .'
def get_requirement(file_path :str):
    requirement=[]
    with open(file_path) as file_obj: 
        requirement=file_obj.readlines()
        requirement=[req.replace("\n","") for req in requirement]
        if HYPHEN_E_DOT in requirement: 
            requirement.remove(HYPHEN_E_DOT)
    return requirement
    
    
setup(
    name='StudentPrediction',
    version='0.0.1', 
    author='Arunabh Singh',
    author_email='sarunabh64@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement("requirements.txt")
    
)