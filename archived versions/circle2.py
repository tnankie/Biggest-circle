import os
print(os.getcwd())
print(os.listdir())
       
from math import sqrt
import time
from queue import PriorityQueue
from math import inf



def _point_to_polygon_distance(x, y, polygon):
    inside = False
    min_dist_sq = inf

    for ring in polygon:
        b = ring[-1]
        for a in ring:

            if ((a[1] > y) != (b[1] > y) and
                    (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])):
                inside = not inside

            min_dist_sq = min(min_dist_sq, _get_seg_dist_sq(x, y, a, b))
            b = a

    result = sqrt(min_dist_sq)
    if not inside:
        return -result
    return result


def _get_seg_dist_sq(px, py, a, b):
    x = a[0]
    y = a[1]
    dx = b[0] - x
    dy = b[1] - y

    if dx != 0 or dy != 0:
        t = ((px - x) * dx + (py - y) * dy) / (dx * dx + dy * dy)

        if t > 1:
            x = b[0]
            y = b[1]

        elif t > 0:
            x += dx * t
            y += dy * t

    dx = px - x
    dy = py - y

    return dx * dx + dy * dy


class Cell(object):
    def __init__(self, x, y, h, polygon):
        self.h = h
        self.y = y
        self.x = x
        self.d = _point_to_polygon_distance(x, y, polygon)
        self.max = self.d + self.h * sqrt(2)


def _get_centroid_cell(polygon):
    area = 0
    x = 0
    y = 0
    points = polygon[0]
    b = points[-1]  # prev
    for a in points:
        f = a[0] * b[1] - b[0] * a[1]
        x += (a[0] + b[0]) * f
        y += (a[1] + b[1]) * f
        area += f * 3
        b = a
    if area == 0:
        return Cell(points[0][0], points[0][1], 0, polygon)
    return Cell(x / area, y / area, 0, polygon)

    pass


def polylabel(polygon, precision=1.0, debug=False):
    # find bounding box
    first_item = polygon[0][0]
    min_x = first_item[0]
    min_y = first_item[1]
    max_x = first_item[0]
    max_y = first_item[1]
    for p in polygon[0]:
        if p[0] < min_x:
            min_x = p[0]
        if p[1] < min_y:
            min_y = p[1]
        if p[0] > max_x:
            max_x = p[0]
        if p[1] > max_y:
            max_y = p[1]

    width = max_x - min_x
    height = max_y - min_y
    cell_size = min(width, height)
    h = cell_size / 2.0

    cell_queue = PriorityQueue()

    if cell_size == 0:
        return [min_x, min_y]

    # cover polygon with initial cells
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            c = Cell(x + h, y + h, h, polygon)
            y += cell_size
            cell_queue.put((-c.max, time.time(), c))
        x += cell_size

    best_cell = _get_centroid_cell(polygon)

    bbox_cell = Cell(min_x + width / 2, min_y + height / 2, 0, polygon)
    if bbox_cell.d > best_cell.d:
        best_cell = bbox_cell

    num_of_probes = cell_queue.qsize()
    while not cell_queue.empty():
        _, __, cell = cell_queue.get()

        if cell.d > best_cell.d:
            best_cell = cell

            if debug:
                print('found best {} after {} probes'.format(
                    round(1e4 * cell.d) / 1e4, num_of_probes))

        if cell.max - best_cell.d <= precision:
            continue

        h = cell.h / 2
        c = Cell(cell.x - h, cell.y - h, h, polygon)
        cell_queue.put((-c.max, time.time(), c))
        c = Cell(cell.x + h, cell.y - h, h, polygon)
        cell_queue.put((-c.max, time.time(), c))
        c = Cell(cell.x - h, cell.y + h, h, polygon)
        cell_queue.put((-c.max, time.time(), c))
        c = Cell(cell.x + h, cell.y + h, h, polygon)
        cell_queue.put((-c.max, time.time(), c))
        num_of_probes += 4

    if debug:
        print('num probes: {}'.format(num_of_probes))
        print('best distance: {}'.format(best_cell.d))
    bradius = _point_to_polygon_distance(best_cell.x, best_cell.y, polygon)
    return [best_cell.x, best_cell.y, bradius]
import re

with open("ex3_sim.cup",'r') as test:
    coords = []
    count = 0
    for line in test:
        if re.search("\d{4}.*W",line) != None:
            keep = re.search("\d{4}.*W",line)
            keep = keep.group()
            print(keep)
            print(keep[0:2],keep[2:8],keep[10:13], keep[13:19])
            keep = [int(keep[0:2])+float(keep[2:8])/60,int(keep[10:13])+float(keep[13:19])/60]
            print(keep)
            coords.append(keep)

        
        
##        if len(line) >24:
##            if count < 50:
##                print(line[0])
##                print('v check ', line[24])
##                print('coord check ', line[7:24])
##            if line[0] == 'B':
##                if line[24] =='A':
##                    coords.append(line[7:24])
##                    if count < 50:
##                        print('wrote')
        count = count +1
    print(len(coords))
    
    cleaned = []
    for line in coords:
        
        cleaned.append(line)
    print(len(cleaned))
    print('coords ', coords[5])
    print('cleaned ', cleaned[5])
    cleaned.append(cleaned[-1])
    print(len(cleaned))
    print(polylabel([cleaned]))
    print('done')
    
