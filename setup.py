from setuptools import setup, find_packages

setup(
    name='sys_id_tools',
    version='0.1',
    description = 'some tools for systems identification from experimental data',
    author='Will Dickson',
    author_email='wbd@caltech',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    packages=find_packages(exclude=['examples',]),
)
