import pytest


# Just a simple test that pytest is working alright


def test():
    with pytest.raises(TypeError):
        raise TypeError("This should not be raised ;)")
