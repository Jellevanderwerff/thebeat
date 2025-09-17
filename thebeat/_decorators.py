# Copyright (C) 2022-2023  Jelle van der Werff
#
# This file is part of thebeat.
#
# thebeat is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# thebeat is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with thebeat.  If not, see <https://www.gnu.org/licenses/>.

import functools
import os
import shutil


def requires_lilypond(f):
    @functools.wraps(f)
    def requires_lilypond_wrapper(*args, **kwds):
        try:
            import lilypond
        except ImportError:
            lilypond = None

        if lilypond is None and not shutil.which('lilypond'):
            raise ImportError("This function or method requires lilypond for plotting notes. You can install this "
                              "optional dependency with pip install 'thebeat[music-notation]'.\n"
                              "Note that if you're on Mac Silicon (so an M1/M2/etc. chip), you will need to install "
                              "lilypond manually. \n"
                              "For more details, see https://thebeat.readthedocs.io/en/latest/installation.html.")

        if lilypond is not None:
            orig_path = os.environ["PATH"]
            os.environ["PATH"] = os.path.dirname(lilypond.executable()) + os.pathsep + os.environ["PATH"]
        try:
            return f(*args, **kwds)
        finally:
            if lilypond is not None:
                os.environ["PATH"] = orig_path

    return requires_lilypond_wrapper
