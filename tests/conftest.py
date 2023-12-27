# Copyright (C) 2023  Jelle van der Werff
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

import pytest


# Remove once a new version of pytest-mpl is released, and brings --mpl-default-style as CLI and/or INI option
# See also https://github.com/matplotlib/pytest-mpl/issues/198
def pytest_configure(config):
    pytest.custom_mpl_image_compare = pytest.mark.mpl_image_compare(backend='agg', style='default', tolerance=0)
