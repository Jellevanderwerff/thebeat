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

        if not lilypond and not shutil.which('lilypond'):
            raise ImportError("This function or method requires lilypond for plotting notes. You can install this "
                              "opional depencency with pip install thebeat[music_notation].\n"
                              "For more details, see https://thebeat.readthedocs.io/en/latest/installation.html.")
        orig_path = os.environ["PATH"]
        os.environ["PATH"] += os.pathsep + os.path.dirname(lilypond.executable())
        return_value = f(*args, **kwds)
        os.environ["PATH"] = orig_path
        return return_value

    return requires_lilypond_wrapper
