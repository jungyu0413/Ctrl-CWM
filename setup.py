from setuptools import setup, find_packages

setup(
    name="ctrl_cwm",
    version="0.1.0",
    description="Ctrl-CWM: Controllable Crowd Generation through World-Model Planning",
    packages=find_packages(include=["ctrl_cwm", "ctrl_cwm.*"]),
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy", "pandas", "scipy", "pillow", "pyyaml"],
)
