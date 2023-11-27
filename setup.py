from setuptools import setup, find_packages
import codecs
import os
import pathlib

VERSION = '0.0.1'
DESCRIPTION = 'A package for compressing video recorded in wildlife monitoring camera trap devices. '

setup(
    name="EcoMotionZip",
    version=VERSION,
    author="Malika Nisal Ratnayake",
    author_email="<malika.ratnayake@monash.edu>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description =  pathlib.Path("README.md").read_text(),
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    project_urls={
        "GitHub": "https://github.com/malikaratnayake/EcoMotionZip.git"
        },
    keywords=['python', 'video', 'stream', 'video stream', 'camera stream', 'sockets'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Ecologists",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Raspberry Pi",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows"
    ],
    python_requires='>=3.10, <3.12',
    install_requires=['requests', 'opencv-python >= 4.8'],
    entry_points={
        'console_scripts': ['EcoMotionZip = EcoMotionZip.__main__:main']
    },

)