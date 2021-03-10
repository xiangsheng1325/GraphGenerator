import setuptools
from setuptools.command.develop import develop
import os
import subprocess
BASEPATH = os.path.dirname(os.path.abspath(__file__))


class CustomDevelop(develop):
    def run(self):
        original_cwd = os.getcwd()

        folders = [
            os.path.join(BASEPATH, 'GraphGenerator/models/bigg/tree_clib'),
        ]
        for folder in folders:
            os.chdir(folder)
            subprocess.check_call(['make'])

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
