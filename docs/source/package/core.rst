Most important classes
======================

There are three classes that make up the core of this package:

* The :py:class:`Sequence<combio.core.Sequence>` class, which only contains **timing** information.
* The :py:class:`Stimulus<combio.core.Sequence>` class, which only contains a **sound**.
* The :py:class:`StimSequence<combio.core.Sequence>` class, which contains **both timing information and sounds**.



Sequence class
--------------

This is the most important class. Check out some functions below:

.. autoclass:: combio.core.Sequence
    :members:
    :undoc-members:

Stimulus class
--------------

The Stimulus class holds an auditory stimulus. 

.. autoclass:: combio.core.Stimulus
    :members:
    :undoc-members:

StimSequence class
------------------

The StimSequence class is a combination of the sound of a Stimulus (or stimuli) and the timing information from a Sequence. 

.. autoclass:: combio.core.StimSequence
    :members:
    :undoc-members:

References
----------
.. footbibliography::