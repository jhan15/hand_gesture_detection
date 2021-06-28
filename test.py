import numpy as np
from sympy import Plane, Point3D

def calculate_angle(vec1, vec2, vec3):
    """ Calculate angle of <vec1,vec2,vec3> """
    plane = Plane(Point3D(vec2), Point3D(vec1), Point3D(vec3))
    print(plane)
    normal_vector = np.array(plane.normal_vector, dtype=int)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    print(normal_vector)
    
    vec21 = vec1 - vec2
    vec23 = vec3 - vec2
    dot = np.dot(vec21, vec23)
    cross = np.cross(vec23, vec21)

    normed_cross = cross / np.linalg.norm(cross)
    print(normed_cross)
    x = np.dot(normal_vector, normed_cross)
    print('x =', x)

    angle = np.arctan2(np.dot(cross, normal_vector), dot)

    if angle < 0:
        angle += 2 * np.pi

    return angle

v1 = np.array([3,0,3])
v2 = np.array([3,0,0])
# v3 = np.array([4,-1,-2])
v3 = np.array([0,4,-2])

angle = calculate_angle(v1,v2,v3)
print(angle)