"""A test runner counts only as the command word of a shell command."""

import pytest

from trajlens.adapters.generic import classify_tool


@pytest.mark.parametrize(
    "command",
    [
        "pip install pytest",
        "grep -rn pytest .",
        "echo pytest",
        "which pytest",
        "cat pytest.ini",
    ],
)
def test_mentions_are_not_test_runs(command):
    assert classify_tool("bash", command) != "test"


def test_find_file_is_search():
    assert classify_tool("find_file", "find_file tox.ini") == "search"
    assert classify_tool("search_dir", 'search_dir "pytest"') == "search"
    assert classify_tool("search_file", "search_file pytest a.py") == "search"


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "cd /repo && python -m pytest tests/",
        "PYTHONPATH=. timeout 300 pytest -x",
        "uv run pytest",
        "npx jest",
        ".venv/bin/pytest",
        "ls; pytest",
        "make build | tee log && make test",
        'bash -c "cd /r && pytest"',
        "python manage.py test auth",
        "./manage.py test",
        "./tests/runtests.py admin_views",
        "python tests/runtests.py --settings=test_sqlite",
        "python -m unittest discover",
        "coverage run -m pytest",
        "go test ./...",
        "cargo test",
        "npm run test",
    ],
)
def test_runner_as_command_word(command):
    assert classify_tool("bash", command) == "test"
