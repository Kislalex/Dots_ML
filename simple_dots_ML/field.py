import sys
import random
import math
import time
import numpy as np
from shapely import geometry as gm


def line_intersections(polygon, line):
    polygon_ = gm.Polygon(polygon)
    points = polygon_.intersection(line)
    if points.is_empty:
        return []
    return list(points.coords)


def turn_unit_vector_by_angle(vect, angle):
    turn_matrix = np.array(
        [np.cos(angle), np.sin(angle), -np.sin(angle), np.cos(angle)]
    ).reshape(2, 2)
    res = np.dot(turn_matrix, vect)
    res = res / np.linalg.norm(res)
    return res


def compute_azimut(target, pos, vel):
    # if we actually have non-zero velocity
    if np.linalg.norm(vel) < 0.001:
        return 0
    normal_ = vel / np.linalg.norm(vel)
    direction = target - pos
    if np.linalg.norm(direction) < 0.001:
        return 0
    direction = direction / np.linalg.norm(direction)
    alpha = np.arctan2(np.cross(direction, normal_), np.dot(direction, normal_))
    return alpha


def closet_point(polygon, pos):
    polygon_exterior = gm.LinearRing(polygon.exterior.coords)
    point = gm.Point(pos)
    d = polygon_exterior.project(point)
    p = polygon_exterior.interpolate(d)
    return list(p.coords)


class Field:
    dist_to_touch = 5
    vision_depth = 100

    def __init__(self, finish, info_size=2):
        self.finish = np.array(finish)
        self.directions = []
        if info_size > 2:
            self.directions = list(np.linspace(-math.pi / 2, math.pi / 2, info_size))[
                1:-1
            ]
        self.obsticales = []
        self.checkpoints = []

    def add_obsticale(self, polygon):
        self.obsticales.append(gm.Polygon(polygon))

    def add_checkpoint(self, polygon):
        self.checkpoints.append(gm.Polygon(polygon))

    def is_close_to_obsticale(self, pos):
        for obs in self.obsticales:
            dist_ = obs.distance(gm.Point(pos))
            if dist_ < self.dist_to_touch:
                return True
        return False

    def is_passing_checkpoint(self, pos, check_num):
        if check_num >= len(self.checkpoints):
            return self.is_finished(pos)
        dist_ = self.checkpoints[check_num].distance(gm.Point(pos))
        return dist_ < self.dist_to_touch

    def is_finished(self, pos):
        dist_ = np.linalg.norm(self.finish - pos)
        return dist_ < 2 * self.dist_to_touch

    def find_next_goal(self, pos, vel, check_num):
        # if the next goal is final - return direction to it
        if check_num >= len(self.checkpoints):
            return [
                np.linalg.norm(self.finish - pos) / 1000.0,
                compute_azimut(self.finish, pos, vel) / 7.0,
            ]
        else:
            point = closet_point(self.checkpoints[check_num], pos)
            return [
                np.linalg.norm(point - pos) / 1000.0,
                compute_azimut(point, pos, vel) / 7.0,
            ]

    def gather_info(self, pos, vel, check_num):
        goal_info = self.find_next_goal(pos, vel, check_num)
        vision_info = self.dot_obsticales_vision(pos, vel)
        return np.array(goal_info + vision_info, dtype=float)

    def dot_obsticales_vision(self, pos, vel):
        if np.linalg.norm(vel) < 0.001:
            vel = self.finish - pos

        vision_lines = []
        for direction in self.directions:
            ray = turn_unit_vector_by_angle(vel, direction)
            # set up the ray line in direction
            vision_lines.append(gm.LineString([pos, pos + self.vision_depth * ray]))

        gm_pos = gm.Point(pos)
        vision_distances = []
        for line in vision_lines:
            # find closes point of intersection
            current_vision = self.vision_depth
            # print(current_vision)
            for obs in self.obsticales:
                points = obs.intersection(line)
                if not points.is_empty:
                    current_vision = min(current_vision, points.distance(gm_pos))
            vision_distances.append(current_vision / self.vision_depth)
        return vision_distances
