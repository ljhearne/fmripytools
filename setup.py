from setuptools import setup, find_packages

setup(
    name='fmripytools',
    version='0.1',
    description='Python based fMRI tools',
    author='LJH',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'nilearn',
        'nibabel',
        'numpy',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'denoise = fmripytools.denoise:main',
            'parcellate = fmripytools.parcellate:main',
            'estimate_fc = fmripytools.estimate_fc:main'
        ],
    }
)
