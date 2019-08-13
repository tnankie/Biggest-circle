import os
print(os.getcwd())
print(os.listdir())
import re       
from math import sqrt
import time
from queue import PriorityQueue
from math import inf
import matplotlib.pyplot as plt
import numpy as np
from math import pi

def is_left(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

def wn_PnPoly(P, V):
    wn = 0   # the winding number counter

    # repeat the first vertex at end
##    V = tuple(V[:]) + (V[0],)

    # loop through all edges of the polygon
    for i in range(len(V)-1):     # edge from V[i] to V[i+1]
        if V[i][1] <= P[1]:        # start y <= P[1]
            if V[i+1][1] > P[1]:     # an upward crossing
                if is_left(V[i], V[i+1], P) > 0: # P left of edge
                    wn += 1           # have a valid up intersect
        else:                      # start y > P[1] (no test needed)
            if V[i+1][1] <= P[1]:    # a downward crossing
                if is_left(V[i], V[i+1], P) < 0: # P right of edge
                    wn -= 1           # have a valid down intersect
    print('point is ', P)
    
    print('wn is ', wn)

    return wn



def _point_to_polygon_distance(x, y, polygon):
    inside = False
    min_dist_sq = inf

    for ring in polygon:
        b = ring[-1]
        for a in ring:

##            if ((a[1] > y) != (b[1] > y) and
##                    (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])):
##                inside = not inside
            
            
            min_dist_sq = min(min_dist_sq, _get_seg_dist_sq(x, y, a, b))
            b = a

    result = sqrt(min_dist_sq)
    point = [x,y]
    if wn_PnPoly(point,polygon) == 0:
                inside = not inside
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


def polylabel(polygon, precision=.1, debug=False):
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

with open("example4.igc",'r') as test:
    coords = []
    count = 0
    for line in test:
        if len(line) >24:
            if count < 50:
                print(line[0])
                print('v check ', line[24])
                print('coord check ', line[7:24])
            if line[0] == 'B':
                if line[24] =='A':
                    coords.append(line[7:24])
                    if count < 50:
                        print('wrote')
        count = count +1
    print(len(coords))
    
    cleaned = []
    count = 0
    for line in coords:
        filler = [float(line[0:2])+float(line[2:7])/60000, float(line[8:11])+float(line[11:16])/60000]
        cleaned.append(filler)
        # if count <5:
        #     print(line)
        #     print(line[0:2])
        #     print(line[2:7])
        #     print(float(line[2:7])/60000)
        #     print(line[8:11])
        #     print(line[11:16])
        #     print(float(line[11:16])/60000)
        #     print(filler)
        count = count +1
    print(len(cleaned))
    print('coords ', coords[49])
    print('cleaned ', cleaned[49])
    cleaned.append(cleaned[-1])
    print(len(cleaned))
    print(cleaned[1])

    x=[]
    y=[]
    x2=[]
    y2=[]

    for line in cleaned:
        x.append(line[1])
        y.append(line[0])
    
    for line in cleaned:
        x2.append(line[1])
        y2.append(line[0])
    
    
    # plt.scatter(x,y, c='b', marker='x', label='1')
    

    # ax1.scatter(x2,y2, c='r', marker='s', label='-1')
    # plt.legend(loc='upper left')
    # plt.show()
    results = polylabel([cleaned])
    print(results)
    radius = results[2]*(110.567 + (results[0]/90 * (111.699 - 110.567)))
    print('circle radius is ', radius,'km ', 'circumfrence is ', 2*pi*radius,'km')
    def xy(h,k,r,phi):
        return h + r*np.cos(phi), k + r*np.sin(phi)

    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')  

    phis=np.arange(0,6.28,0.01)
    ax.scatter(x,y, c='b', s=1, label='1')
    r =results[2]
    ax.plot( *xy(results[1],results[0],r,phis), c='r',ls='-' )
    bill = plt.gca()
    bill.set_ylim(results[0]-1.1*results[2],results[0]+1.1*results[2])
    bill.set_xlim(results[1]-1.1*results[2],results[1]+1.1*results[2])
    # plt.show()
    plt.savefig('precis01.png')
    # circle1 = plt.Circle((results[0],results[1]), results[2], color='r')
    # plt.gcf().gca().add_artist(circle1)
    # plt.show()
     
    print('done')
    
