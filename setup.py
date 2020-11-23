"""
setup script
"""
from setuptools import setup,find_packages
import multigrids

config = {
    'description': 'Multigrids',
    'author': 'Rawser Spicer',
    'url': multigrids.__metadata__.__url__,
    'download_url': multigrids.__metadata__.__url__,
    'author_email': 'rwspicer@alaska.edu',
    'version': multigrids.__metadata__.__version__,
    'install_requires': ['numpy','pyyaml', 'moviepy'],
    'packages': find_packages(),
    'scripts': [],
    'package_data': {},
    'name': 'Multigrids'
}

setup(**config)
