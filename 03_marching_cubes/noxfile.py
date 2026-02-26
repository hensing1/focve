import nox

nox.options.sessions = []


@nox.session
def tests(session):
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest", "tests")

if __name__ == "__main__":
    nox.main()
