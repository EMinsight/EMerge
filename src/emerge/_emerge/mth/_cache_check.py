# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

from numba.core import event
from numba import njit

_COMPILE_MESSAGE = """
 [ EMERGE ]
⚠  Numba will be compiling optimized code in this first run; this may take a few minutes.
   • Additional functions may be compiled on-the-fly.
   • Compilation happens only once-subsequent runs load from cache.
   Please wait…"""

@njit(cache=True)
def _donothing(a):
    return a

class Notify(event.Listener):
    def on_start(self, ev):
        f = ev.data['dispatcher']
        sig = ev.data['args']
        if f is _donothing:          # limit to the function you care about
            sig = ev.data['args']
            print(_COMPILE_MESSAGE)

    def on_end(self, ev):        # unused here
        pass


# install listener only for this block:
with event.install_listener("numba:compile", Notify()):
    _donothing(0) 