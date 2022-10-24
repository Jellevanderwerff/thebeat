Most important classes
======================

There are three classes that make up the core of this package:

* The :py:class:`Sequence<thebeat.core.Sequence>` class, which only contains **timing** information.
* The :py:class:`Stimulus<thebeat.core.Sequence>` class, which only contains a **sound**.
* The :py:class:`StimSequence<thebeat.core.Sequence>` class, which contains **both timing information and sounds**.

In addition, there is the :py:class:`BaseSequence<thebeat.core.BaseSequence>` class, 
which is the parent class of the :py:class:`Sequence<thebeat.core.Sequence>`, :py:class:`Rhythm<thebeat.rhythm.Rhythm>`, 
and :py:class:`StimSequence<thebeat.core.StimSequence>` classes. 
In principle it is irrelevant, except for understanding the internal workings of this package.

.. toctree:: 
    :hidden:

    Sequence class<core/Sequence.rst>
    Stimulus class<core/Stimulus.rst>
    StimSequence class<core/StimSequence.rst>
    BaseSequence class<core/BaseSequence.rst>
