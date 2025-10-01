import pytest
from onorm import Normalizer


def test_instantiate():
    with pytest.raises(TypeError):
        Normalizer()
