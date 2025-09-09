import numpy as np

class Region:
    def __init__(self, filename):
        raise Exception("You cannot instantiate the pure base class Region")
        
    def check_inside_absolute(self, x, y):
        """Return True if (x, y) is inside the region"""
        raise Exception("You cannot instantiate the pure base class Region")
    
class BoxRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("box("):
                self.success = False
                return
            x, y, l, w, angle = line[4:-2].split(",")
            if ":" in x:
                raise Exception("Can only load `ciao` formatted region files in physical coordinates")
            self.x = float(x)
            self.y = float(y)
            self.l = float(l)
            self.w = float(w)
            self.angle = float(angle) * np.pi / 180
            self.success = True

    def get_alpha(self, x, y, v0, v1):
        return ((x - self.x) * v0 + (y - self.y) * v1) / (v0*v0 + v1*v1)
    
    def check_inside_absolute(self, x, y):
        # Assume x and y are in degrees
        l_alpha = self.get_alpha(x, y, self.l * np.cos(self.angle), self.l * np.sin(self.angle))
        w_alpha = self.get_alpha(x, y, self.w * np.sin(self.angle), -self.w * np.cos(self.angle))
        return (np.abs(l_alpha) < 0.5) & (np.abs(w_alpha) < 0.5)

    
class PolygonRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("polygon("):
                self.success = False
                return
            points = line[8:-2].split(",")
            self.points = []
            for i in range(0, len(points), 2):
                self.points.append((float(points[i]), float(points[i+1])))
            self.points = np.array(self.points)
            self.success = True

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


    
class CircleRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("circle("):
                self.success = False
                return
            x, y, radius = line[7:-2].split(",")
            self.x = float(x)
            self.y = float(y)
            self.radius2 = float(radius)**2
            self.success = True
    
    def check_inside_absolute(self, x, y):
        dist2 = (x - self.x)**2 + (y - self.y)**2
        return dist2 < self.radius2
    

    
class EllipseRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("ellipse("):
                self.success = False
                return
            ra, dec, a, b, angle = line[8:-2].split(",")
            self.ra = float(ra)
            self.dec = float(dec)
            self.a = float(a)
            self.b = float(b)
            self.angle = float(angle) * np.pi / 180 - np.pi/2
            self.success = True
    
    def check_inside_absolute(self, x, y):
        rot_x = np.sin(self.angle) * (x - self.ra) + np.cos(self.angle) * (y - self.dec)
        rot_y = -np.cos(self.angle) * (x - self.ra) + np.sin(self.angle) * (y - self.dec)
        d = rot_x**2 / self.a**2 + rot_y**2 / self.b**2
        return d < 1
    
def make_region(region_file):
    reg = BoxRegion(region_file)
    if reg.success: return reg
    reg = PolygonRegion(region_file)
    if reg.success: return reg
    reg = CircleRegion(region_file)
    if reg.success: return reg
    reg = EllipseRegion(region_file)
    if reg.success: return reg
    raise Exception("Region not implemented")

