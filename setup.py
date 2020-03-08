from setuptools import setup#,find_packages

setup(
    #packages=find_packages(),
    name='eva',
    version='0.0.3',
    description='My private package from private EVA repo',
    url='git@github.com:ojhajayant/eva.git',
    author='Jayant Ojha',
    author_email='ojhajayant@yahoo.com',
    license='unlicense',
    packages=['eva', 'eva.week7', 'eva.week7.modular'],
    zip_safe=False
)
