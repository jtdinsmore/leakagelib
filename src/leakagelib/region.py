import numpy as np

class Region:
    """
    Class to store DS9 regions.
    
    Notes
    -----
    - All regions are assumed to be stored in `ciao` format.
    - All coordinates must be physical
    """
    def __init__(self, filename):
        """
        If you are looking for the region constructor, use `Region.load` instead. 
        """
        raise Exception("You cannot instantiate the pure base class Region")
    
    def load(filename):
        """
        Load a region from a region file

        Parameters
        ----------
        filename : str
            File name of the region

        Returns
        -------
        Region
            Region object
        """
        with open(filename) as f:
            line = f.readline()
            typ = line[:line.find('(')]
        if typ == "circle":
            return CircleRegion(filename)
        elif typ == "polygon":
            return PolygonRegion(filename)
        elif typ == "box":
            return BoxRegion(filename)
        elif typ == "ellipse":
            return EllipseRegion(filename)
        else:
            raise Exception(f"Unrecognized region type `{typ}`")
        
    def check_inside_absolute(self, x, y):
        """
        Return True if (x, y) is inside the region
        """
        raise Exception("You cannot instantiate the pure base class Region")
    
    def area(self):
        """
        Get the area of the region. Area is only defined for circular, elliptical, and box regions
        """
        raise Exception("You cannot instantiate the pure base class Region")
    
class BoxRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("box("):
                raise Exception("Box regions must start with box")
            x, y, l, w, angle = line[4:-2].split(",")
            if ":" in x:
                raise Exception("Can only load `ciao` formatted region files in physical coordinates")
            self.x = float(x)
            self.y = float(y)
            self.l = float(l)
            self.w = float(w)
            self.angle = float(angle) * np.pi / 180

    def get_alpha(self, x, y, v0, v1):
        return ((x - self.x) * v0 + (y - self.y) * v1) / (v0*v0 + v1*v1)
    
    def check_inside_absolute(self, x, y):
        # Assume x and y are in degrees
        l_alpha = self.get_alpha(x, y, self.l * np.cos(self.angle), self.l * np.sin(self.angle))
        w_alpha = self.get_alpha(x, y, self.w * np.sin(self.angle), -self.w * np.cos(self.angle))
        return (np.abs(l_alpha) < 0.5) & (np.abs(w_alpha) < 0.5)

    def area(self):
        return self.l * self.w
    
    
class PolygonRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("polygon("):
                raise Exception("Polygon regions must start with polygon")
            points = line[8:-2].split(",")
            self.points = []
            for i in range(0, len(points), 2):
                self.points.append((float(points[i]), float(points[i+1])))
            self.points = np.array(self.points)

    def check_inside_single(self, x, y):
        # Assume x and y are in degrees
        for line_index in range(len(self.points)):
            intersections_before = 1 # There is a self-intersection
            intersections_after = 0
            for other_line_index in range(len(self.points)):
                if other_line_index == line_index: continue
                # Check if the line intersects this line segment
                mid = (self.points[line_index] + self.points[line_index - 1])/2
                vx = x - mid[0]
                vy = y - mid[1]
                px, py = self.points[other_line_index-1] - self.points[other_line_index]
                diffx, diffy = mid - self.points[other_line_index]
                det = (vx * py - vy * px)
                alpha = (vx * diffy - vy * diffx) / det
                if 0 <= alpha <= 1:
                    beta = (px * diffy - py * diffx) / det
                    if beta > 1:
                        intersections_after += 1
                    else:
                        intersections_before += 1
            if intersections_before % 2 == 0:
                return False
            if intersections_after % 2 == 0:
                return False
        return True
    
    def check_inside_absolute(self, x, y):
        if type(x) == float:
            return self.check_inside_single(x, y)
        else:
            output = np.zeros(len(x)).astype(bool)
            for i, (xi, yi) in enumerate(zip(x, y)):
                output[i] = self.check_inside_single(xi, yi)
            return output

    def area(self):
        area = 0
        for i in range(len(self.points)):
            base = self.points[i][0] - self.points[i-1][0]
            height = (self.points[i][1] + self.points[i-1][1]) / 2
            area += base * height
        return np.abs(area)
    
class CircleRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("circle("):
                raise Exception("Circle regions must start with circle")
            x, y, radius = line[7:-2].split(",")
            self.x = float(x)
            self.y = float(y)
            self.radius2 = float(radius)**2
    
    def check_inside_absolute(self, x, y):
        dist2 = (x - self.x)**2 + (y - self.y)**2
        return dist2 < self.radius2
    
    def area(self):
        return self.radius2 * np.pi
    
    
class EllipseRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("ellipse("):
                raise Exception("Ellipse regions must start with ellipse")
            ra, dec, a, b, angle = line[8:-2].split(",")
            self.ra = float(ra)
            self.dec = float(dec)
            self.a = float(a)
            self.b = float(b)
            self.angle = float(angle) * np.pi / 180 - np.pi/2
    
    def check_inside_absolute(self, x, y):
        rot_x = np.sin(self.angle) * (x - self.ra) - np.cos(self.angle) * (y - self.dec)
        rot_y = np.cos(self.angle) * (x - self.ra) + np.sin(self.angle) * (y - self.dec)
        d = rot_x**2 / self.a**2 + rot_y**2 / self.b**2
        return d < 1
    
    def area(self):
        return self.a * self.b * np.pi