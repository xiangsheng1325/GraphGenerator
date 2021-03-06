import setuptools
from setuptools.command.develop import develop
import os
import subprocess
BASEPATH = os.path.dirname(os.path.abspath(__file__))


class CustomDevelop(develop):
    def run(self):
        original_cwd = os.getcwd()
        folder = os.path.join(BASEPATH, 'GraphGenerator/models/kronecker_ops')
        if not os.path.exists(os.path.join(folder, 'Makefile.config')):
            os.chdir(folder)
            subprocess.check_call(['unzip', '-o', '-d', '.', 'kronecker_src.zip'])
        folders = [
            os.path.join(BASEPATH, 'GraphGenerator/models/bigg_ops/tree_clib'),
            os.path.join(BASEPATH, 'GraphGenerator/models/kronecker_ops/examples/kronfit')
        ]
        for folder in folders:
            os.chdir(folder)
            subprocess.check_call(['make'])
        folders = [
            os.path.join(BASEPATH, 'GraphGenerator/evaluate'),
        ]
        for folder in folders:
            os.chdir(folder)
            subprocess.check_call(['g++', '-O2', '-std=c++11', '-o', 'orca', 'orca.cpp'])
        os.chdir(original_cwd)

        super().run()


setuptools.setup(
    name="GraphGenerator",
    version="0.1",
    author="Sheng Xiang",
    author_email="xiangsheng218@gmail.com",
    description="Graph Generator package",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    cmdclass={
        'develop': CustomDevelop
    }
)
