Installation guide
==================

The WelDX package can be installed using *conda* or *mamba* package manager from the :code:`conda-forge` channel. These
managers originate from the freely available `Anaconda Python stack <https://docs.conda.io/en/latest/miniconda.html>`_.
If you do not have Anaconda or Miniconda installed yet, we ask you to install *Miniconda*-3. Documentation for the
installation procedure can be
found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_.

After this step you have access to the conda command and can proceed to installing the WelDX package::

    conda create -n weldx -c conda-forge weldx

The package is also available on pypi and can be installed via *pip*::

    pip install weldx

Setting up Jupyter Lab
----------------------

Weldx provides lots of visualization methods for planning and analysis. These methods need a frontend like
Jupyter lab or Jupyter notebook. We currently recommend to use Jupyter lab, as it is modern and makes working with
several notebooks easier. You can install Jupyter lab both via *conda* or *pip*.
If you use conda we suggest that you create a separate environment for your weldx installation and jupyter.
This keeps the environments clean and easier to upgrade (is that really true? think of mixed versions of extensions in lab env and weldx env!).

Here is a guide on howto setup different kernels for
Jupyter `guide <https://ipython.readthedocs.io/en/7.25.0/install/kernel_install.html>`_.


Create an environment named "jupyter" via conda::

    conda create -n jlab jupyter-lab -c conda-forge

Then we switch to the weldx environment created in the first step and make it available within Jupyter::

    conda activate weldx
    python -m ipykernel install --user --name weldx --display-name "Python (weldx)"

This will enable us to select the Python interpreter installed in the weldx environment within Jupyter. So when a new
notebook is created, we can choose "Python (weldx)" to access all the software bundled with the weldx Python package.

Build and enable Jupyter lab extensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We need to install several different extensions for Jupyter::

    conda activate jlab
    jupyter labextension install @jupyter-widgets/jupyterlab-manager k3d


Everything in one-shot
----------------------
If you feel lucky, you can try and copy-paste all install commands into a shell. Note that if one command fails,
all subsequent commands will not be executed.

using conda::

    conda create -n weldx -c conda-forge weldx
    conda activate weldx
    python -m ipykernel install --user --name weldx --display-name "Python (weldx)"
    conda create -n jlab -c conda-forge jupyter-lab
    conda activate jlab
    jupyter labextension install @jupyter-widgets/jupyterlab-manager k3d
