import os
from setuptools import setup

setup(
    name = "sandbox",
    version = "0.1",
    author = "Charanpal Dhanjal ",
    author_email = "charanpal@gmail.com",
    description = ("An demonstration of how to create, document, and publish "
                                   "to the cheese shop a5 pypi.org."),
    license = "GPLv3",
    keywords = "numpy",
    url = "http://packages.python.org/sandbox",
    packages=['sandbox.centering', 'sandbox.clustering', 'sandbox.util'],
    long_description="A collection of machine learning algorithms",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License"
    ],
)
