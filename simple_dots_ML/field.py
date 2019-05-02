import sys
import random
import math
import time
import numpy as np
from shapely import geometry as gm

def lineIntersections(polygon, line):
    polygon_ = gm.Polygon(polygon)
    points = polygon_.intersection(line)
    if (points.is_empty):
        return []
    return list(points.coords) 

def turnByAngleAndNormailze(vect, angle):
    turn = np.array([[ np.cos(angle),  np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]])
    res = np.dot(turn, vect)
    res = np.divide(res, np.linalg.norm(res))
    return res

def computeDirections(point, pos, vel):
    dist_ = np.linalg.norm(np.subtract(point, pos)) / 2000.0
    alpha = 0
    #if we actually have non-zero velocity
    if (np.linalg.norm(vel) > 0.001):
        r = np.subtract(point, pos)
        r = np.divide(r, np.linalg.norm(r))
        n = np.divide(vel, np.linalg.norm(vel))
        # find the angle between
        alpha = np.arctan2(np.cross(r, n), np.dot(r,n)) / 10.0
    return [dist_, alpha]

def closetPoint(polynom, pos):
    pol_ext = gm.LinearRing(polynom.exterior.coords)
    point = gm.Point(pos)
    d = pol_ext.project(point)
    p = pol_ext.interpolate(d)
    return list(p.coords)


class Field:
    
    dist_to_touch = 5
    vision_depth = 100
    
    def __init__(self, finish, info_size = 2):
        self.finish = np.array(finish)
        self.directions = []
        if (info_size > 2):
            self.directions = list(np.linspace(-math.pi / 2, math.pi / 2, info_size))[1:-1]
        self.obsticales = []
        self.checkpoints = []

    def addObsticale(self, polygon):
        self.obsticales.append(gm.Polygon(polygon))

    def addCheckPoint(self, polygon):
        self.checkpoints.append(gm.Polygon(polygon))
        
    def isCloseToObsticale(self, pos):
        for obs in self.obsticales:
            dist_ = obs.distance(gm.Point(pos))
            if (dist_ < self.dist_to_touch):
                return True
        return False

    def isPassingCheckPoint(self, pos, check_num):
        if (check_num >= len(self.checkpoints)):
            return self.isFinished(pos)
        dist_ = self.checkpoints[check_num].distance(gm.Point(pos))
        return (dist_ < self.dist_to_touch)

    def isFinished(self, pos):
        dist_ = np.linalg.norm(np.subtract(self.finish, pos))
        return (dist_ < 2 * self.dist_to_touch)

    def findNextGoal(self, pos, vel, check_num):
        #if the next goal is final - return direction to it
        if (check_num >= len(self.checkpoints)):
            return computeDirections(self.finish, pos, vel)
        else :
            point = closetPoint(self.checkpoints[check_num], pos)
            return computeDirections(point, pos, vel)
        
    def gatherInfo(self, pos, vel, check_num):
        goal_info = self.findNextGoal(pos, vel, check_num)
        vision_info = self.pointObservation(pos, vel)
        #return np.array(computeDirections(self.finish, pos, vel) + goal_info + vision_info)
        return np.array(goal_info + vision_info)

    
    def pointObservation(self, pos, vel):
        res = []
        if (np.linalg.norm(vel) < 0.001):
                vel = np.subtract(self.finish, pos)
        vision_lines = []
        for dir_ in self.directions:
            ray = turnByAngleAndNormailze(vel, dir_)
            # set up the ray line in direction
            vision_lines.append(gm.LineString([pos, np.add(pos, self.vision_depth * ray)]))
        pos_point = gm.Point(pos)
        for line in vision_lines:
            # find closes point of intersection
            current_vision = self.vision_depth
            #print(current_vision)
            for obs in self.obsticales:
                points = obs.intersection(line)
                if (not points.is_empty):
                    current_vision = min(current_vision, points.distance(pos_point))
            res.append(current_vision / self.vision_depth)
        return res
        

