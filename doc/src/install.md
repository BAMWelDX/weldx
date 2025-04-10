# Installation guide

The WelDX package can be installed using any conda or mamba package manager from the [Conda-Forge channel](https://conda-forge.org/#about).
If you have not yet installed a conda package manager, we recommend installing `Miniforge`.
The installer can then be found [here](https://conda-forge.org/download/), and a detailed documentation for the installation process is provided
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).
Once this step has been completed, you will gain access to both the `conda` and the `mamba` command and will be able to proceed with the installation of the WelDX package.

If you freshly installed `Miniforge`, open the installed `Anaconda Prompt` command line interface
and run:

```shell
conda init
mamba init
```

to initialize `conda` and `mamba` as commands in all your shells.

In order to create a new conda environment called `weldx` containing the WeldX package,
run the console command:

```console
conda create --name weldx --channel conda-forge weldx weldx_widgets
```

To install the WeldX package into your existing environment instead, use:

```shell
conda install weldx weldx_widgets --channel conda-forge
```

If installed, all `conda` commands can be replaced by `mamba` to take advantage
of its faster solver.

The package is also available on pypi and can be installed via:

```shell
pip install weldx weldx-widgets
```

As weldx currently depends on the package `bottleneck`, which contains
C/C++ code, you will need a working C/C++ compiler. The conda package
does not have this requirement as it only installs pre-compiled
binaries. So if you do not know how to install a working compiler, we
strongly encourage using the conda package.

## Setting up Jupyter Lab

Weldx provides many visualisation methods for planning and analysis.
These methods require a frontend such as JupyterLab or Jupyter notebook.
We recommend using JupyterLab because it is modern and makes it easy to
work with several notebooks. You can install JupyterLab using either conda, mamba,
or pip. If you use conda, we recommend that you create a separate environment
for your weldx installation and Jupyter. This will keep the environments
clean and easier to upgrade.

Create an environment named `jlab` via conda that installs `jupyterlab` and the `k3d` extension:

```shell
conda create --name jlab --channel conda-forge jupyterlab k3d
```

Then we switch to the weldx environment created in the first step and
make it available within Jupyter:

```shell
conda activate weldx
python -m ipykernel install --user --name weldx --display-name "Python (weldx)"
```

This will enable us to select the Python interpreter installed in the
weldx environment within Jupyter. So when a new notebook is created, we
can choose `Python (weldx)` to access all the software bundled with the
weldx Python package.

If you wish to setup multiple different kernels for Jupyter a guide can be found
[here](https://ipython.readthedocs.io/en/7.25.0/install/kernel_install.html).

## Starting the Jupyer Lab

To start JupyterLab run:

```shell
conda activate jlab
jupyter lab
```

A window in your browser will automatically be opened.

## Fixing DLL errors on Windows systems

In case you run into an error when using the weldx kernel on Windows
systems that fails to read DLLs like:

```shell
ImportError: DLL load failed: The specified module could not be found
```

you might have to apply the fix mentioned [here](https://github.com/jupyter/notebook/issues/4569#issuecomment-609901011).

Go to `%userprofile%\.ipython\profile_default\startup` using the
windows explorer and create a new file called `ipython_startup.py`.
Open it with a text editor and paste the following commands into the
file:

```python
import sys
import os
from pathlib import Path


# get directory of virtual environment
p_env = Path(sys.executable).parent

# directory which should contain all the DLLs
p_dlls = p_env / "Library" / "bin"

# effectively prepend this DLL directory to $PATH
# semi-colon used here as sep on Windows
os.environ["PATH"] = "{};{}".format(p_dlls, os.environ["PATH"])
```

## Everything in one-shot

If you feel lucky, you can try and copy-paste all install commands into
a shell. Note that if one command fails, all subsequent commands will
not be executed.

```shell
conda create --name weldx --channel conda-forge weldx weldx_widgets
conda create --name jlab jupyterlab k3d --channel conda-forge
conda activate weldx
python -m ipykernel install --user --name weldx --display-name "Python (weldx)"
conda activate jlab
jupyter lab
```
