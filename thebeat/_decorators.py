import functools
import shutil


def requires_lilypond(f):
    @functools.wraps(f)
    def requires_lilypond_wrapper(*args, **kwds):
        if not shutil.which('lilypond'):
            raise ImportError("This function or method requires lilypond for plotting notes. "
                              "Check out https://lilypond.org/download.en.html for instructions on how to install. "
                              "Make sure to also follow the instructions on how to add lilypond to your PATH.")

        return f(*args, **kwds)
    return requires_lilypond_wrapper
