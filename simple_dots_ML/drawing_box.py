#!/usr/bin/python3.7
import cv2
import os
import time
import math
import numpy as np

from population import Population
from brain.brain import Brain
from field import Field
from field import turn_by_angle_and_normalize as turn_by_angle


height = 900
width = 1200
blue = (255, 0, 0)
orange = (0, 165, 255)
green = (0, 128, 0)
red = (0, 0, 255)
yellow = (140, 230, 240)
ttl = 1
directions = list(np.linspace(-math.pi / 2, math.pi / 2, 7))[1:-1]


def create_fields(num):
    fields = []
    s = "fields/field"
    n = 0
    while os.path.isfile(s + str(n) + ".txt"):
        f = open(s + str(n) + ".txt", "r")
        finish_ = tuple(map(int, f.readline().split(" ")))
        field = Field(finish_, num)
        while True:
            line = list(map(int, f.readline().split(" ")))
            polygon = list(zip(line[::2], line[1::2]))
            if len(polygon) < 2:
                break
            field.add_obsticale(polygon)
        while True:
            line = list(map(int, f.readline().split(" ")))
            polygon = list(zip(line[::2], line[1::2]))
            if len(polygon) < 2:
                break
            field.add_checkpoint(polygon)
        n = n + 1
        fields.append(field)
    return fields


def draw_leader_info(img, leader, field, score):
    cv2.circle(img, tuple(leader.pos.astype(int)), 5, green, -1)
    cv2.circle(img, tuple(leader.pos.astype(int)), 5, (0, 0, 0), 1)
    if np.linalg.norm(leader.vel) > 0.001:
        data = field.gather_info(leader.pos, leader.vel, leader.checkpoints)
        alpha = data[1] * 10.0
        dist_ = data[0] * 2000
        dir_ = turn_by_angle(leader.vel, alpha)
        goal = np.add(leader.pos, dist_ * dir_)
        cv2.circle(img, tuple(goal.astype(int)), 5, orange, -1)
        for i in range(len(directions)):
            alpha = directions[i]
            dist_ = data[i + 2] * 100
            dir_ = turn_by_angle(leader.vel, alpha)
            goal = np.add(leader.pos, dist_ * dir_)
            cv2.line(
                img,
                tuple(leader.pos.astype(int)),
                tuple(goal.astype(int)),
                (0, 0, 0),
                1,
            )
            cv2.circle(img, tuple(goal.astype(int)), 3, orange, -1)
    cv2.putText(img, "max_score: " + str(score), (10, 110), 16, 0.6, (0, 0, 0))


def redraw(blank, fields, population):
    level = 14
    attempt = 1
    max_score = 0
    while True:
        x = population.are_all_dots_stopped()
        if x == 0:
            if population.dots[0].reached_goal:
                level += 1
                if level >= len(fields):
                    level = 0
                max_score = 0
                attempt = 0
            max_score = population.compute_score(fields[level])
            population.natural_selection()
            population.mutation()
            attempt += 1
        else:
            begin = time.time()
            population.update(fields[level])
            end = time.time()
            img = blank.copy()
            for polygon in fields[level].obsticales:
                cv2.polylines(
                    img, [np.array(polygon.exterior.coords, np.int32)], True, blue, 5
                )
            for polygon in fields[level].checkpoints:
                cv2.polylines(
                    img, [np.array(polygon.exterior.coords, np.int32)], True, yellow, 1
                )

            cv2.circle(img, tuple(fields[level].finish.astype(int)), 10, red, -1)
            cv2.circle(img, tuple(fields[level].finish.astype(int)), 10, (0, 0, 0), 1)

            for dot in population.dots[1:]:
                cv2.circle(img, tuple(dot.pos.astype(int)), 3, (0, 0, 0), -1)

            draw_leader_info(img, population.dots[0], fields[level], max_score)

            cv2.putText(
                img, "gen : " + str(population.gen), (10, 30), 16, 0.6, (0, 0, 0), 1
            )
            cv2.putText(
                img, "fps : " + str(1.0 / (end - begin)), (10, 50), 16, 0.6, (0, 0, 0)
            )
            cv2.putText(img, "alive : " + str(x), (10, 70), 16, 0.6, (0, 0, 0))
            cv2.putText(
                img,
                "level : " + str(level) + " attempt : " + str(attempt),
                (10, 90),
                16,
                0.6,
                (0, 0, 0),
            )
            cv2.imshow("ML", img)
            key = cv2.waitKey(ttl)
            if key == ord("q"):
                s = "test_brain.txt"
                f = open(s, "w")
                population.dots[0].dot_brain.save_to_file(f)
                break



blank = 255 * np.ones((height, width, 3), dtype="uint8")
cv2.imshow("ML", blank)

rule = (7, 50, 12, 50, 10, 2)

fields = create_fields(rule[0])
population = Population(np.array([50, height - 50]), 200, rule)
dot_brain_file = open("test_brain.txt", "r")
population.dots[0].dot_brain.read_from_file(dot_brain_file)
dot_brain_file.close()
for i in range(1,200):
    population.dots[i].dot_brain = population.dots[0].dot_brain.copy() 
    population.dots[i].dot_brain.mutate(0.01, 0.03)
#population = Population(np.array([50, height - 50]), 200, rule)
redraw(blank, fields, population)

cv2.destroyAllWindows()
