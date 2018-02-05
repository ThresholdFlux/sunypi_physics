from setuptools import setup

setup(
    name='sunypi_physics',
    version='0',
    packages=['models.kronig_penney.kronig_penney', 'utilities.find_roots', 'utilities.rank_nullspace', 'constants'],
    install_requires=['numpy', 'scipy', 'matplotlib'],
    url='https://github.com/ThresholdFlux/sunypi_physics',
    license='',
    author='Daniel Landers',
    author_email='dans.landers@gmail.com',
    description=''
)
