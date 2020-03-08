from setuptools import setup, find_packages

setup(
    name='eva',
    version='0.0.5',
    description='My private package from private EVA repo',
    url='git@github.com:ojhajayant/eva.git', 
    author='Jayant Ojha',
    author_email='ojhajayant@yahoo.com',
    license='unlicense',
    #package_dir={'': 'eva'}, 
    packages=find_packages(),
    #packages=['week7', 'week7.modular'],
    zip_safe=False
)
