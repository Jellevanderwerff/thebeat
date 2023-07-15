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
import importlib
import shutil


def requires_lilypond(f):
    @functools.wraps(f)
    def requires_lilypond_wrapper(*args, **kwds):
        if not importlib.util.find_spec('lilypond') and not shutil.which('lilypond'):
            raise ImportError("This function or method requires lilypond for plotting notes. "
                              "Check out https://lilypond.org/download.en.html for instructions on how to install. "
                              "Make sure to also follow the instructions on how to add lilypond to your PATH.")

        return f(*args, **kwds)
    return requires_lilypond_wrapper
