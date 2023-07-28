import nox


@nox.session(python=3.10)
def clean(session):
    session.install("black", "nb-clean", "isort")
    session.run(
        "black",
    )
