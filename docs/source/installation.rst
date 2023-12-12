============
Installation
============

.. Hint::

    All methods below require you to have ``pip`` installed. Check if ``pip`` is installed by typing the following in a terminal window.
    If working in a `virtual environment <https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv>`_, make sure you see ``(venv)`` before your username in your terminal:

    .. code-block:: console

        pip --version


    If you do not see a version number, please follow `these <https://pip.pypa.io/en/stable/installation/>`_ instructions to install pip.



*************************************
Installing through PyPI (recommended)
*************************************

*thebeat* is available through PyPI, and can be installed using:

.. code-block:: console

    pip install thebeat

Note that if you want use thebeat's functionality for plotting musical notation,
you have to install it using:

.. code-block:: console

    pip install 'thebeat[music_notation]'

This will install thebeat with the dependencies needed for plotting musical notation.
These dependencies are `abjad <https://abjad.github.io>`_ and `Lilypond <https://lilypond.org>`_.

You can now import *thebeat* in your preferred editor using:

.. code-block:: python

    import thebeat


******************************************
Installing development version from GitHub
******************************************

Open up a terminal and run:

.. code-block:: console

    pip install git+https://github.com/jellevanderwerff/thebeat.git@main#egg=thebeat
