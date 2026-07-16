from setuptools import setup, find_packages

def get_requirements(file_path: str) -> list[str]:
    with open(file_path) as f:
        requirements = f.read().splitlines()
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name = 'Chestnut',
    version = '1.0',
    author= 'Ritesh',
    author_email= 'gupta.2002@outlook.com',
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)