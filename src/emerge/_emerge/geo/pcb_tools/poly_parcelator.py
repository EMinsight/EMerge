import numpy as np
from typing import Generator

from emsutil import Material
import matplotlib.pyplot as plt

DSMIN = 1e-6
def get_line_segment_intersection(p1, p2, pl1, pl2) -> tuple[tuple[float, float], float]:
    """
    p1, p2: (x, y) coordinates of the segment
    pl1, pl2: (x, y) coordinates defining the infinite line
    Returns (x, y) if intersection exists, else None
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = pl1
    x4, y4 = pl2

    # Determinant of the system
    # Line 1 (Segment): P = p1 + t(p2 - p1)
    # Line 2 (Infinite): P = pl1 + u(pl2 - pl1)
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # If denominator is 0, lines are parallel
    if abs(denom) < 1e-10:
        return None
    
    # Intersection point parameters
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    # Check if the intersection point lies within the segment [0, 1]
    if 0 <= t <= 1:
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return (intersect_x, intersect_y), t *((x2-x1)**2 + (y2-y1)**2)**0.5
    
    return None

class Cutline:

    def __init__(self, p0: tuple[float, float], p1: tuple[float, float]):
        self.p0: tuple[float, float] = p0
        self.p1: tuple[float, float] = p1

class CutlineSet:

    def __init__(self):
        self.lineset: list[Cutline] = []

    def add_line(self, p0: tuple[float, float], p1: tuple[float, float]):
        self.lineset.append(Cutline(p0,p1))


class PCBPolygon:

    def __init__(self, xs: list[float], ys: list[float], z: float, material: Material):
        self.xs: list[float] = xs
        self.ys: list[float] = ys
        self.z: float = z
        self.material: Material = material

    def iter_edges(self) -> Generator[tuple[tuple[float, float], tuple[float,float]], None, None]:
        n = len(self.xs)
        for i1 in range(n):
            yield (self.xs[i1], self.ys[i1]), (self.xs[(i1+1)%n], self.ys[(i1+1)%n])

    @property
    def xys(self) -> list[tuple[float, float]]:
        return list(zip(self.xs, self.ys))
    
    def simplify(self):
        i = 0
        
        DSMINSQ = DSMIN**2
        while i < len(self.xs):
            n = len(self.xs)
            i += 1
            dist = ((self.xs[i % n] - self.xs[i-1])**2 + (self.ys[i % n] - self.ys[i-1])**2)
            if dist < DSMINSQ:
                self.xs.pop(i % n)
                self.ys.pop(i % n)
                i -= 1
            
    def fragment(self, cutlines: CutlineSet) -> None:
        newxs = []
        newys = []
        n = len(self.xs)
        for iseg in range(n):
            x0 = self.xs[iseg]
            y0 = self.ys[iseg]
            x1 = self.xs[(iseg+1)%n]
            y1 = self.ys[(iseg+1)%n]
            newxs.append(x0)
            newys.append(y0)
            cutpts = []
            for line in cutlines.lineset:
                out = get_line_segment_intersection((x0,y0), (x1,y1), line.p0, line.p1)
                if out is not None:
                    cutpts.append(out)
            
            cutpts = sorted(cutpts, key=lambda x: x[1])
            for ((xn, yn), t) in cutpts:
                if (t > DSMIN) and ((xn-x0)**2 + (yn-y0)**2)**0.5 > DSMIN and ((xn-x1)**2 + (yn-x1)**2)**0.5 > DSMIN:
                    newxs.append(xn)
                    newys.append(yn)
                    
        self.xs = newxs
        self.ys = newys
        self.simplify()

class PolygonSet:

    def __init__(self):
        self.polies: list[PCBPolygon] = []
        self.cutlines: CutlineSet = CutlineSet()

    def add_poly(self, xys: list[tuple[float, float]], z: float, material: Material) -> None:
        xs = [x[0] for x in xys]
        ys = [y[1] for y in xys]
        self.polies.append(PCBPolygon(xs, ys, z, material))
    
    def generate_cutlines(self) -> None:
        for poly in self.polies:
            for p1, p2 in poly.iter_edges():
                dx = p2[0]-p1[0]
                dy = p2[1]-p1[1]
                pn1 = (p1[0] - dy, p1[1] + dx)
                pn2 = (p2[0] - dy, p2[1] + dx)
                self.cutlines.add_line(p1,p2)
                self.cutlines.add_line(p1, pn1)
                self.cutlines.add_line(p2, pn2)
        
    def fragment(self):
        for poly in self.polies:
            poly.fragment(self.cutlines)
        #self.plot_polygon_set()

    def plot_polygon_set(self, title="Polygon Set Visualization"):
        """
        Plots all polygons in a PolygonSet with nodes clearly marked.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for i, poly in enumerate(self.polies):
            # Close the polygon loop for plotting by appending the first point to the end
            plot_xs = poly.xs + [poly.xs[0]]
            plot_ys = poly.ys + [poly.ys[0]]
            
            # Plot the perimeter
            ax.plot(plot_xs, plot_ys, marker='o', linestyle='-', label=f'Poly {i}')
            
            # Plot the nodes explicitly
            ax.scatter(poly.xs, poly.ys, color='red', zorder=5)

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

    