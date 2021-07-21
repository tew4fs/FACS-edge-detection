import pytest
from edgedetection.example import Example

class TestClass:
    def testOne(self):
        x = "this"
        assert "h" in x

    def testTwo(self):
        ex = Example()
        x = ex.example()
        assert x == "this"