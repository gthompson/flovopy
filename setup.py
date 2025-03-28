from setuptools import setup, find_packages

setup(
    name='flovopy',
    version='0.1.0',
    description='Seismology and volcano observatory tools including Seisan parsing',
    author='Your Name',
    author_email='you@example.com',
    url='https://github.com/yourusername/flovopy',  # Optional
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'obspy',
        'pandas',
        'numpy',
        # add any others
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.8',
)
