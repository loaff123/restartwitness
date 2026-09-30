# Dependency and source inventory

Original code, equations, fixture data, SVG reports and documentation: BSD-3-Clause. No third-party source, binaries, assets or datasets are bundled.

Runtime: Python standard library (Python license); NumPy 2.x (BSD-3-Clause). Development: pytest (MIT), Hypothesis (MPL-2.0), build (MIT), setuptools (MIT), wheel (MIT). Exact local versions are in requirements-lock.txt; this is a tested environment record, not a claim all future versions are compatible.

Optional, separately installed:
- REBOUND4.4.11, GPL-3.0-or-later. [Source/license](https://github.com/hannorein/rebound)
- OpenMM8.4.0.post2. API, application and Reference/CPU components have MIT licensing; GPU platform terms differ (including LGPL). This package selects only Reference. [Official licensing](https://docs.openmm.org/latest/userguide/library/01_introduction.html)

Check each dependency's current license when redistributing a combined application. Installing optional libraries is not the same as bundling their code under RestartWitness's license. No commercial service is required.
