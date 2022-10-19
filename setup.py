from setuptools import setup

setup(
    name="Eye of Beholder",
    packages=["orangeext"],
    package_data={"orangeext": ["icons/*.svg"]},
    classifiers=["Example :: Invalid"],
    entry_points={"orange.widgets": "Demo = orangeext"},
)
