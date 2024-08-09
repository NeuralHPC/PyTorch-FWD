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
    session.run("isort", "tests", "noxfile.py")
    session.run("black", "tests", "noxfile.py")


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
    session.install(".")
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
        "--explicit-package-bases",
        "src",
    )


@nox.session(name="test")
def test(session):
    """Run long pytest."""
    session.install(".")
    session.chdir("tests")
    # env handles deterministic CuBLAS with CUDA >= 10.2
    session.run("pytest")


@nox.session(name="fast-test")
def run_test_fast(session):
    """Run pytest."""
    session.install(".")
    session.install("pytest")
    # env handles deterministic CuBLAS with CUDA >= 10.2
    session.run("pytest", "-m", "not slow")


@nox.session(name="build")
def build(session):
    """Build a pip package."""
    session.install("build")
    session.run("python", "-m", "build")


@nox.session(name="finish")
def finish(session):
    """Finish this version increase the version number and upload to pypi.
    To upload push the version tag. Git push origin tag <tag_name>.
    """
    session.run("bumpversion", "release", external=True)


@nox.session(name="new-patch")
def finish(session):
    """Start a new patch version."""
    session.run("bumpversion", "patch", external=True)


@nox.session(name="check-package")
def pyroma(session):
    """Run pyroma to check if the package is ok."""
    session.install("pyroma")
    session.run("pyroma", ".")
