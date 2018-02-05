from setuptools import setup

setup(
    name='sunypi_physics',
    version='0',
    packages=['models.kronig_penney', 'utilities.find_roots', 'utilities.rank_nullspace', 'constants'],
    install_requires=['numpy', 'scipy', 'matplotlib'],
    url='',
    license='',
    author='Daniel Landers',
    author_email='dans.landers@gmail.com',
    description=''
)
