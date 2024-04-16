"""Noxfile."""

import nox


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "noxfile.py")
    session.run("black", "src", "noxfile.py")
    session.run("isort", "scripts", "noxfile.py")
    session.run("black", "scripts", "noxfile.py")


@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("flake8==4.0.1")
    session.install(
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.run("flake8", "src", "tests", "noxfile.py")


@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install("-r", "requirements.txt")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--no-strict-optional",
        "--no-warn-return-any",
        "--implicit-reexport",
        "--allow-untyped-calls",
        "src",
    )


@nox.session(name="test")
def test(session):
    """Run long pytest."""
    session.install("-r", "requirements.txt")
    session.chdir("tests")
    session.run("pytest")


@nox.session(name="fast-test")
def run_test_fast(session):
    """Run pytest."""
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest", "-m", "not slow")
