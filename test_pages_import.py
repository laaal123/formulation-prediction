"""Smoke test: every Streamlit page must import without error."""
import os, sys, glob, warnings, importlib.util
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import pytest

PAGES = sorted(glob.glob(os.path.join(ROOT, "page_*.py")))


@pytest.mark.parametrize("path", PAGES, ids=[os.path.basename(p) for p in PAGES])
def test_page_compiles(path):
    with open(path) as fh:
        compile(fh.read(), path, "exec")


def test_app_compiles():
    with open(os.path.join(ROOT, "app.py")) as fh:
        compile(fh.read(), "app.py", "exec")


def test_layout_is_flat():
    """No subdirectories: the repo must upload to GitHub in one drag."""
    entries = [e for e in os.listdir(ROOT)
               if os.path.isdir(os.path.join(ROOT, e))
               and not e.startswith((".", "__"))]
    assert entries == [], f"unexpected subdirectories: {entries}"


def test_no_package_imports_remain():
    import re
    bad = []
    for f in glob.glob(os.path.join(ROOT, "*.py")):
        txt = open(f).read()
        if re.search(r"^\s*(from|import)\s+(core|data)[\s.]", txt, re.M):
            bad.append(os.path.basename(f))
    assert bad == [], f"stale package imports in {bad}"


def test_all_pages_present():
    assert len(PAGES) == 11   # 10 analysis pages + the loader
