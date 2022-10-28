from setuptools import setup

NAME = "Eye Of Beholder"
VERSION = "0.0.1"
AUTHOR = 'Ji Qi'
AUTHOR_EMAIL = 'j.qi@esciencecenter.nl'
URL = 'https://github.com/EyeofBeholder-NLeSC/orange-addon'
DESCRIPTION = "Add-on developed for the eye of the beholder project."
LICENSE = "Apache License 2.0"
PACKAGES = ['orangeext']
INSTALL_REQUIRES = [
    'Orange3',
    'csvw'
]
ENTRY_POINTS = {'orange.widgets': ('Eye Of Beholder = orangeext')}


if __name__ == '__main__':
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        description=DESCRIPTION,
        license=LICENSE,
        packages=PACKAGES,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
    )