# /// script
# dependencies = [
#   "nox",
#   "numpy",
#   "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git",
#   "Pillow",
#   "polyscope",
#   "pytest",
#   "torch",
#   "trimesh",
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
