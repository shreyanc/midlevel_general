import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="midlevel",
    version="1.1.0",
    author='Shreyan Chowdhury',            
    author_email='shreyan0311@gmail.com',
    description='Package to compute mid-level features from audio',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
)

