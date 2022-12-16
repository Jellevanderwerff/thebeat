Most important classes
======================

There are three classes that make up the core of this package:

* The :py:class:`Sequence<thebeat.core.Sequence>` class, which only contains **timing** information.
* The :py:class:`SoundStimulus<thebeat.core.SoundStimulus>` class, which only contains a **sound**.
* The :py:class:`SoundSequence<thebeat.core.SoundSequence>` class, which contains **both timing information and sounds**.

In addition, there is the :py:class:`BaseSequence<thebeat.core.BaseSequence>` class, 
which is the parent class of the :py:class:`Sequence<thebeat.core.Sequence>`, :py:class:`Rhythm<thebeat.rhythm.Rhythm>`, 
and :py:class:`SoundSequence<thebeat.core.SoundSequence>` classes. 
In principle it is irrelevant, except for understanding the internal workings of this package.

.. toctree:: 
    :hidden:

    Sequence class<core/Sequence.rst>
    SoundStimulus class<core/SoundStimulus.rst>
    SoundSequence class<core/SoundSequence.rst>
    BaseSequence class<core/BaseSequence.rst>
