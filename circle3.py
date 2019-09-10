import os
print(os.getcwd())
print(os.listdir())
import re
import math
from math import sqrt
import time
from queue import PriorityQueue
from math import inf
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import sys

# curvature of the earth constants used to define the wgs84 elipsoid
a = 6378137
f = 1/298.257223563
b = a*(1-f)
e = ((a**2 - b**2)/a**2)**0.5
eprime = ((a**2 - b**2)/b**2)**0.5

def xyz(phi,lam,h):
    phi = phi*pi/180
    lam = lam*pi/180
    N = a/((1-((e**2)*math.sin(phi)**2))**.5)

    X = (N + h)*math.cos(phi)*math.cos(lam)
    Y = (N + h)*math.cos(phi)*math.sin(lam)
    Z = (((b**2/a**2)*N) +h)*math.sin(phi)
    return(X,Y,Z)
#used to bugcheck the xyz function
##print(xyz(0,0,0))
##print(xyz(90,180,0))
##
##print('other side ', xyz(-90,-180,0))

def is_left(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

def wn_PnPoly(P, V):
    wn = 0   # the winding number counter
    #print('now in the wn function, here is the point', P)
    #print(V)
    # repeat the first vertex at end
    # V = tuple(V[:]) + (V[0],)
    #print(V)


    # loop through all edges of the polygon
    #print('length of V ', len(V))
    #print('type of V:', type(V))
    #print(V[0])
    #print('type of V[0]:', type(V[0]))
    V = V[0]
    #print('length of V ', len(V))
    count = 0
    for i in range(len(V)-1):     # edge from V[i] to V[i+1]
        count= count + 1
        if V[i][1] <= P[1]:        # start y <= P[1]
            if V[i+1][1] > P[1]:     # an upward crossing
                if is_left(V[i], V[i+1], P) > 0: # P left of edge
                    #print('crossed upward')
                    wn += 1           # have a valid up intersect
        else:                      # start y > P[1] (no test needed)
            if V[i+1][1] <= P[1]:    # a downward crossing
                if is_left(V[i], V[i+1], P) < 0: # P right of edge
                    wn -= 1           # have a valid down intersect
                    #print('crossed downward')
##    print('point is ', P)
##  
    #print('count is ', count)  
    #print('wn is ', wn)

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
    #print(result)
    if wn_PnPoly(point, polygon) == 0:
                #print('outside')
                inside = not inside
                result = -result
    #print(result)
 
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
    #print('centroid cell')
    #print(x / area, y / area, area)
    return Cell(x / area, y / area, 0, polygon)

    pass


def polylabel(polygon, precision=0.1, debug=True):
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
    print('min max X then Y')
    print(min_x, max_x)
    print(min_y, max_y)
    width = max_x - min_x
    height = max_y - min_y
    print('height, width, h')
    cell_size = min(width, height)
    divisor = 2
    h = cell_size / divisor
    print(height, width, h)
    cell_queue = PriorityQueue()

    if cell_size == 0:
        return [min_x, min_y]

    # cover polygon with initial cells
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            c = Cell(x + h, y + h, h, polygon)
            y += h
            cell_queue.put((-c.max, time.time(), c))
        x += h

    best_cell = _get_centroid_cell(polygon)
    #print(type(best_cell), best_cell.h, best_cell.x, best_cell.y)

    bbox_cell = Cell(min_x + width / divisor, min_y + height / divisor, 0, polygon)
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

        h = cell.h / divisor
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


def get_poly(filename):
    poly =[]
    with open(filename) as test:
        coords = []
        count = 0
        for line in test:
            if len(line) > 24:
                if line[0] == 'B':
                    if line[24] == 'A':
                        coords.append(line[7:])
##        print(len(coords))
        coords.append(coords[0])
##        print(len(coords))
##        print('last point  ', coords[-2])
##        print('first point ',coords[-1])
        cleaned = []
        
        #converting to decimal degrees 
        for line in coords:
            lat = int(line[0:2]) + float(line[2:7])/60000
            alt = int(line[23:28])
            r_lat =lat
            if line[7] == 'S':
                r_lat = 0-lat
            long = int(line[8:11]) + float(line[11:16])/60000
            r_long = long
            if line[16] == 'W':
                r_long = 0-long
            #bug checking for correct indexes for conversion
##            if count <3:
##                print(lat,line[2:7],float(line[2:7]),line[7])
##                print(long,line[11:16],float(line[11:16]),line[16])
##                print(line[23:28])
            position = [r_lat, r_long, alt]
            cleaned.append(position)
            count = count + 1
        print('last point  ', cleaned[-2])
        print('first point ', cleaned[-1])    
        
        
        #find average position
        sum_lat = 0
        for i in cleaned:
            sum_lat = sum_lat +i[0]
        ave_lat = sum_lat/len(cleaned)

        sum_long = 0
        for i in cleaned:
            sum_long = sum_long +i[1]
        ave_long = sum_long/len(cleaned)

        sum_alt = 0
        for i in cleaned:
            sum_alt = sum_alt +i[2]
        ave_alt = sum_alt/len(cleaned)

        ave_pos = [ave_lat, ave_long, ave_alt]
        print("Average position ", ave_lat, ave_long, ave_alt)

        
        #convert to xyz from polar co-ordinates
        euclidian = []
        for line in cleaned:
            euclidian.append(xyz(line[0],line[1],line[2]))
##        print('last point  ', euclidian[-2])
##        print('first point ', euclidian[-1])

        #build 2D plane
        #generate normal vector centered on average of flight log
        vec = xyz(ave_lat, ave_long, ave_alt)
        sum_sqr_vec = ((vec[0]**2) + (vec[1]**2) + (vec[2]**2))**0.5
        n_vec = [vec[0]/sum_sqr_vec, vec[1]/sum_sqr_vec, vec[2]/sum_sqr_vec]        
##        print(vec, n_vec)
        print(n_vec)
        #find smallest component and zero, flip remaining, negate one
        abs_n_vec = []
        indexes = [0,1,2]
        for i in n_vec:
            abs_n_vec.append(abs(i))
##        print(abs_n_vec)
        smallest = abs_n_vec.index(min(abs_n_vec))
        print(smallest)
        print(indexes)
        indexes.pop(smallest)
        print(indexes)
        help_vec = [0, 0, 0]
        help_vec[indexes[0]] = -1*n_vec[indexes[1]]
        help_vec[indexes[1]] = n_vec[indexes[0]]
        print(help_vec)
        axis_a = np.cross(n_vec, help_vec)
        axis_b = np.cross(n_vec, axis_a)
        print(axis_a, axis_b)

        #test orthongonality of axis a,b,normal
        print(np.dot(n_vec, axis_a), np.dot(n_vec, axis_b), np.dot(axis_b, axis_a))
        #transform euclidean points
        poly = []
        for i in euclidian:
            dif = [0,0,0]
            for b in range(3):
                dif[b] = i[b] - vec[b]
            
            t1 = np.dot(axis_a, dif)
            t2 = np.dot(axis_b, dif)
            poly.append([t1, t2])
        print(poly[0])
        print(len(poly), len(cleaned))
            
        
        
        
    return(poly, axis_a, axis_b)

pos_files = os.listdir()
good_files = []

for file in pos_files:
##    print(file[-4:]) # error checking file name extension
    if file[-4:] == '.igc':
        good_files.append(file)

print(good_files)
poly_axis = get_poly(good_files[0])
print(len(poly_axis[0]))
beef = poly_axis[0]
print(len(beef))

results  = polylabel([beef])
print(results)
radius = results[2]
print('circle radius is ', radius,'m ', 'circumfrence is ', 2*pi*radius,'m')
x=[]
y=[]
x2=[]
y2=[]

for line in beef:
    x.append(line[1])
    y.append(line[0])
    
for line in beef:
    x2.append(line[1])
    y2.append(line[0])

plt.scatter(x,y, c='b', marker='x', label='1')
#ax1.scatter(x2,y2, c='r', marker='s', label='-1')
plt.legend(loc='upper left')
plt.show()
def xy(h,k,r,phi):
    return h + r*np.cos(phi), k + r*np.sin(phi)

    
# figure plotting to verify and visualise results
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')  

phis=np.arange(0,6.28,0.01)
ax.scatter(x,y, c='b', s=1, label='1')
r =results[2]
ax.plot( *xy(results[1],results[0],r,phis), c='r',ls='-' )
bill = plt.gca()
bill.set_ylim(results[0]-1.1*results[2],results[0]+1.1*results[2])
bill.set_xlim(results[1]-1.1*results[2],results[1]+1.1*results[2])
plt.show()
plt.savefig('precis01_euclid.png')
circle1 = plt.Circle((results[0],results[1]), results[2], color='r')
plt.gcf().gca().add_artist(circle1)
#plt.show()
     
print('done')



##with open("ex3_sim.cup",'r') as test:
##    coords = []
##    count = 0
##    for line in test:
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
##        count = count +1
##    print(len(coords))
##    
##    cleaned = []
##    count = 0
##    for line in coords:
##        filler = [float(line[0:2])+float(line[2:7])/60000, float(line[8:11])+float(line[11:16])/60000]
##        cleaned.append(filler)
##        # if count <5:
##        #     print(line)
##        #     print(line[0:2])
##        #     print(line[2:7])
##        #     print(float(line[2:7])/60000)
##        #     print(line[8:11])
##        #     print(line[11:16])
##        #     print(float(line[11:16])/60000)
##        #     print(filler)
##        count = count +1
####    print(len(cleaned))
####    print('coords ', coords[49])
####    print('cleaned ', cleaned[49])
##    cleaned.append(cleaned[-1])
####    print(len(cleaned))
####    print(cleaned[1])
##
##    x=[]
##    y=[]
##    x2=[]
##    y2=[]
##
##    for line in cleaned:
##        x.append(line[1])
##        y.append(line[0])
##    
##    for line in cleaned:
##        x2.append(line[1])
##        y2.append(line[0])
    
    
    # plt.scatter(x,y, c='b', marker='x', label='1')
    

    # ax1.scatter(x2,y2, c='r', marker='s', label='-1')
    # plt.legend(loc='upper left')
    # plt.show()
##    results = polylabel([cleaned])
##    print(results)
##    radius = results[2]*(110.567 + (results[0]/90 * (111.699 - 110.567)))
##    print('circle radius is ', radius,'km ', 'circumfrence is ', 2*pi*radius,'km')
##    def xy(h,k,r,phi):
##        return h + r*np.cos(phi), k + r*np.sin(phi)
##
##    
### figure plotting to verify and visualise results
##    fig = plt.figure()
##    ax = fig.add_subplot(111,aspect='equal')  
##
##    phis=np.arange(0,6.28,0.01)
##    ax.scatter(x,y, c='b', s=1, label='1')
##    r =results[2]
##    ax.plot( *xy(results[1],results[0],r,phis), c='r',ls='-' )
##    bill = plt.gca()
##    bill.set_ylim(results[0]-1.1*results[2],results[0]+1.1*results[2])
##    bill.set_xlim(results[1]-1.1*results[2],results[1]+1.1*results[2])
##    plt.show()
##    plt.savefig('precis01_20.png')
##    # circle1 = plt.Circle((results[0],results[1]), results[2], color='r')
##    # plt.gcf().gca().add_artist(circle1)
##    # plt.show()
##     
##    print('done')
    
