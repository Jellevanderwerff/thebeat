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

    pip install thebeat[musical_notation]

You can now import *thebeat* in your preferred editor using:

.. code-block:: python

    import thebeat


.. Hint::
    For Windows, take a look at `this <https://www.digitalcitizen.life/command-prompt-how-use-basic-commands/>`_ tutorial if you do not know how to navigate to a directory using the command line.
    For Mac OS, take a look at `this <https://www.macworld.com/article/221277/command-line-navigating-files-folders-mac-terminal.html>`_ tutorial.
    For Linux, take a look at `this one <https://www.cyberciti.biz/faq/how-to-change-directory-in-linux-terminal/>`_.



******************************************
Installing development version from GitHub
******************************************

Open up a terminal and run:

.. code-block:: console

    pip install git+https://github.com/jellevanderwerff/thebeat.git#egg=thebeat
