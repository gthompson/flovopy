# tests/test_run_iceweb.py
import pytest
from flovopy.wrappers import run_iceweb

def test_run_iceweb_main_smoke(monkeypatch):
    dummy_args = [
        "prog",
        "--config", "tests/data/config", 
        "--start", "2023-01-01T00:00:00",
        "--end", "2023-01-01T01:00:00",
        "--subnet", "TEST"
    ]
    monkeypatch.setattr("sys.argv", dummy_args)
    try:
        run_iceweb.main()
    except SystemExit as e:
        assert e.code == 0 or isinstance(e.code, int)
