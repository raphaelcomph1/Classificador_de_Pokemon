import pokedex


def test_package_imports() -> None:
    assert pokedex.__version__ == "0.1.0"
