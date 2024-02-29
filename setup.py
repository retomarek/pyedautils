import setuptools
import some_build_toolkit

def get_version():
    version = some_build_toolkit.compute_version()
    return version

setuptools.setup(
    version=get_version(),
)
