============
Installation
============

.. Hint::

    All methods below require you to have ``pip`` installed. Check if ``pip`` is installed by typing the following in a terminal window.
    If working in a `virtual environment <https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv>`_, make sure you see ``(venv)`` before your username in your terminal:

    .. code-block:: console

        pip --version
    
    
    If you do not see a version number, please follow `these <https://pip.pypa.io/en/stable/installation/>`_ instructions to install pip.



***************************
Installing from a .zip file
***************************

Save the ``.zip`` file to a directory that you can access easily from a terminal.

Open up a terminal, and navigate to the directory of the saved ``.zip`` file.

Install *thebeat* using:

.. code-block:: console

    pip install thebeat-main.zip


You can now import *thebeat* in your preferred editor using:

.. code-block:: python

    import thebeat


.. Hint::
    For Windows, take a look at `this <https://www.digitalcitizen.life/command-prompt-how-use-basic-commands/>`_ tutorial if you do not know how to navigate to a directory using the command line.
    For Mac OS, take a look at `this <https://www.macworld.com/article/221277/command-line-navigating-files-folders-mac-terminal.html>`_ tutorial.
    For Linux, take a look at `this one <https://www.cyberciti.biz/faq/how-to-change-directory-in-linux-terminal/>`_




**********************
Easy install from PyPI
**********************

Open up a terminal, and type:

.. code-block:: console

    pip install thebeat

Now, you can import *thebeat* in your code using e.g.:

.. code-block:: python

    import thebeat


******************************************
Installing development version from Github
******************************************

Open up a terminal and run:

.. code-block:: console

    pip install https://github.com/jellevanderwerff/thebeat/thebeat/zipball/dev
