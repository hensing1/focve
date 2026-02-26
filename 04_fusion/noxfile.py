# /// script
# dependencies = [
#   "nox",
#   "numpy",
#   "Pillow",
#   "polyscope",
#   "pytest",
#   "scikit-image",
#   "scipy",
#   "torch",
#   "tqdm",
# ]
# ///

import nox

nox.options.sessions = []


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit tests."""
    session.run("pytest", "tests", external=True)

if __name__ == "__main__":
    nox.main()
