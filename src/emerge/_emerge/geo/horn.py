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

from ..geometry import GeoVolume
from ..cs import CoordinateSystem
from ..selection import FaceSelection

import gmsh # type: ignore

class Horn(GeoVolume):

    def __init__(self,
                 start_aperture: tuple[float, float],
                 end_aperture: tuple[float, float],
                 height: float,
                 cs: CoordinateSystem):
        """Generate a Horn geometry. The horn is defined by a start and ending rectangle.

        Start aperture is always defined at z=0 and the end at z=height.
        The horn can be reoriented by choosing a different coordinate system.

        Args:
            start_aperture (tuple[float, float]): The width/height of the start aperture
            end_aperture (tuple[float, float]): The width/height of the end aperture
            height (float): The height of the horn
            cs (CoordinateSystem): The coordinate systme to place the horn in.
        """
        super().__init__([])
        p0 = cs.origin
        p1 = p0 + cs.zax.np * height

        wax = cs.xax.np
        hax = cs.yax.np
        dax = cs.zax.np

        w1, h1 = start_aperture
        w2, h2 = end_aperture

        p11 = p0 + wax *w1/2 + hax * h1/2
        p12 = p0 + wax *w1/2 - hax * h1/2
        p13 = p0 - wax *w1/2 - hax * h1/2
        p14 = p0 - wax *w1/2 + hax * h1/2

        p21 = p1 + wax *w2/2 + hax * h2/2
        p22 = p1 + wax *w2/2 - hax * h2/2
        p23 = p1 - wax *w2/2 - hax * h2/2
        p24 = p1 - wax *w2/2 + hax * h2/2

        pt11 = gmsh.model.occ.addPoint(*p11)
        pt12 = gmsh.model.occ.addPoint(*p12)
        pt13 = gmsh.model.occ.addPoint(*p13)
        pt14 = gmsh.model.occ.addPoint(*p14)
        pt21 = gmsh.model.occ.addPoint(*p21)
        pt22 = gmsh.model.occ.addPoint(*p22)
        pt23 = gmsh.model.occ.addPoint(*p23)
        pt24 = gmsh.model.occ.addPoint(*p24)

        l1r = gmsh.model.occ.addLine(pt11, pt12)
        l1b = gmsh.model.occ.addLine(pt12, pt13)
        l1l = gmsh.model.occ.addLine(pt13, pt14)
        l1t = gmsh.model.occ.addLine(pt14, pt11)
        
        l2r = gmsh.model.occ.addLine(pt21, pt22)
        l2b = gmsh.model.occ.addLine(pt22, pt23)
        l2l = gmsh.model.occ.addLine(pt23, pt24)
        l2t = gmsh.model.occ.addLine(pt24, pt21)

        dtr = gmsh.model.occ.addLine(pt11, pt21)
        dbr = gmsh.model.occ.addLine(pt12, pt22)
        dbl = gmsh.model.occ.addLine(pt13, pt23)
        dtl = gmsh.model.occ.addLine(pt14, pt24)

        wbot = gmsh.model.occ.addWire([l1r, l1b, l1l, l1t])
        wtop = gmsh.model.occ.addWire([l2r, l2b, l2l, l2t])
        wright = gmsh.model.occ.addWire([l1r, dbr, l2r, dtr])
        wleft = gmsh.model.occ.addWire([l1l, dbl, l2l, dtl])
        wfront = gmsh.model.occ.addWire([l1b, dbl, l2b, dbr])
        wback = gmsh.model.occ.addWire([l1t, dtr, l2t, dtl])

        s1 = gmsh.model.occ.addSurfaceFilling(wbot)
        s2 = gmsh.model.occ.addSurfaceFilling(wtop)
        s3 = gmsh.model.occ.addSurfaceFilling(wright)
        s4 = gmsh.model.occ.addSurfaceFilling(wleft)
        s5 = gmsh.model.occ.addSurfaceFilling(wfront)
        s6 = gmsh.model.occ.addSurfaceFilling(wback)

        sv = gmsh.model.occ.addSurfaceLoop([s1, s2, s3, s4, s5, s6])

        self.tags: list[int] = [gmsh.model.occ.addVolume([sv,]),]
        
        pc = p0 + dax * height/2
        self._add_face_pointer('front', pc - height/2*dax, -dax)
        self._add_face_pointer('back', pc + height/2*dax, dax)
        
    @property
    def front(self) -> FaceSelection:
        """The first local -Z face of the Horn."""
        return self.face('front')
    
    @property
    def back(self) -> FaceSelection:
        """The back local +Z face of the Horn."""
        return self.face('back')
    
    @property
    def sides(self) -> FaceSelection:
        """The outside faces excluding the top and bottom."""
        return self.boundary(exclude=('front','back'))