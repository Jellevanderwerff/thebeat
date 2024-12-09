# Copyright (C) 2024  Jelle van der Werff
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

import sys

import pytest

NUMPY2_DOCTESTS = [
    'thebeat.stats.ks_test'
]


def pytest_collection_modifyitems(items):
    skipif = pytest.mark.skipif(condition=sys.version_info < (3, 9), reason="Doctest requires NumPy 2 to be installed")
    for item in items:
        if item.name in NUMPY2_DOCTESTS:
            item.add_marker(skipif)
