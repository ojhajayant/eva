from setuptools import setup, find_packages

setup(
    name='eva',
    version='0.0.9',
    description='My private package from private EVA repo',
    url='https://github.com/ojhajayant/eva',
    author='Jayant Ojha',
    author_email='ojhajayant@yahoo.com',
    license='unlicense',
    package_dir={'': 'eva'}, 
    packages=find_packages(where='eva'),
    #packages=['week7', 'week7.modular'],
    zip_safe=False
)
