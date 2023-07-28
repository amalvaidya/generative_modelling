import nox


@nox.session(python="3.10")
def clean(session):
    session.install("black", "nb-clean", "isort")
    session.run("black", "sections")
    session.run("nb-clean", "clean", "sections")
    session.run("isort", "sections")
