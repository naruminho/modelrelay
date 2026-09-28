"""A versão é uma só: pyproject.toml e modelrelay.__version__ (o sagadeck exige uma versão mínima por ela)."""
import tomllib
from pathlib import Path

import modelrelay


def test_version_is_the_same_in_pyproject_and_package():
    pyproject = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == modelrelay.__version__
