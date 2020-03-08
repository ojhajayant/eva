from setuptools import setup#,find_packages

setup(
    #packages=find_packages(),
    name='eva',
    version='0.0.6',
    description='My private package from private EVA repo',
    url='git@github.com:ojhajayant/eva.git',
    author='Jayant Ojha',
    author_email='ojhajayant@yahoo.com',
    license='unlicense',
    packages=['week7', 'week7.modular'],
    zip_safe=False
)
