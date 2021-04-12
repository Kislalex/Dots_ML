import cv2
import os
import time
import math
import numpy as np

from population import Population
from NeuralNetwork.brain import Brain
from field import Field
from field import turn_unit_vector_by_angle as turn_by_angle


height = 900
width = 1200
blue = (255, 0, 0)
orange = (0, 165, 255)
green = (0, 128, 0)
red = (0, 0, 255)
yellow = (140, 230, 240)
ttl = 1
directions = list(np.linspace(-math.pi / 2, math.pi / 2, 7))[1:-1]
slow_learn = True
  

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


def draw_dot(img, dot):
    alpha = 0
    if np.linalg.norm(dot.vel) > 0.01:
        normal_ = np.array([1, 0])
        alpha = (
            np.arctan2(np.cross(dot.vel, normal_), np.dot(dot.vel, normal_))
            * 180.0
            / np.pi
        )
    cv2.ellipse(
        img,
        tuple(dot.pos.astype(int)),
        (4, 2),
        -alpha,
        0,
        360,
        (0, 0, 0),
        -1,
    )


def draw_leader_info(img, leader, field):
    alpha = 0
    if np.linalg.norm(leader.vel) > 0.01:
        normal_ = np.array([1, 0])
        alpha = (
            np.arctan2(np.cross(leader.vel, normal_), np.dot(leader.vel, normal_))
            * 180.0
            / np.pi
        )
    cv2.ellipse(img, tuple(leader.pos.astype(int)), (10, 4), -alpha, 0, 360, green, -1)
    cv2.ellipse(
        img, tuple(leader.pos.astype(int)), (10, 4), -alpha, 0, 360, (0, 0, 0), 1
    )
    if np.linalg.norm(leader.vel) > 0.001:
        data = field.gather_info(leader.pos, leader.vel, leader.checkpoints)
        alpha = data[1] * 7.0
        dist_ = data[0] * 1000.0
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


def write_info(img, population, max_score, tl, level, attempt, alive):
    cv2.putText(
        img,
        "level : {}, attempt: {}".format(level,attempt),
        (10, 20),
        16,
        0.6,
        (0, 0, 0),
    )
    cv2.putText(img, "gen: {}".format(population.gen), (10, 40), 16, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "alive: {}".format(alive), (10, 60), 16, 0.6, (0, 0, 0))
    cv2.putText(img, "max_score: {0:.2f}".format(max_score), (10, 80), 16, 0.6, (0, 0, 0))
    cv2.putText(img, "fps: {0:.1f}".format(1.0 / tl), (10, 100), 16, 0.6, (0, 0, 0))
    


def draw_field(field):
    img = blank.copy()
    for polygon in field.obsticales:
        cv2.polylines(img, [np.array(polygon.exterior.coords, np.int32)], True, blue, 5)
    for polygon in field.checkpoints:
        cv2.polylines(
            img, [np.array(polygon.exterior.coords, np.int32)], True, yellow, 1
        )
    cv2.circle(img, tuple(field.finish.astype(int)), 10, red, -1)
    cv2.circle(img, tuple(field.finish.astype(int)), 10, (0, 0, 0), 1)
    return img

def timed_update(population, field):
    begin = time.time()
    population.update(field)
    return time.time() - begin
    
def redraw(blank, fields, population):
    level = 0
    attempt = 1
    max_score = 0
    while True:
        alive = population.count_alive_dots()
        if alive == 0:
            attempt += 1
            if population.dots[0].reached_goal or slow_learn:
                max_score = population.update_score(fields[level])
                level = (level + 1) % len(fields)
                if not slow_learn :
                    attempt = 0
                    population.natural_selection()
                    population.clear_score()
                elif level == 0:
                    print("Evolving")
                    level = 0
                    attempt = 0
                    population.natural_selection()
                    population.clear_score()
            elif not slow_learn:
                max_score = population.update_score(fields[level])
                population.natural_selection()
                population.clear_score()
            else:
                max_score = population.update_score(fields[level])
            population.restart()
        else:
            tl = timed_update(population, fields[level])
            img = draw_field(fields[level])
            for dot in population.dots[1:]:
                draw_dot(img, dot)
            draw_leader_info(img, population.dots[0], fields[level])
            write_info(img, population, max_score, tl, level, attempt, alive)
            cv2.imshow("DotsML", img)
            key = cv2.waitKey(ttl)
            if key == ord("q"):
                s = "test_brain.txt"
                f = open(s, "wb")
                population.dots[0].dot_brain.write_to_stream(f)
                break


blank = 255 * np.ones((height, width, 3), dtype="uint8")
cv2.imshow("DotsML", blank)

rule = (7, 10, 20, 20, 10, 2)

fields = create_fields(rule[0])
population = Population(np.array([50, height - 50]), 100, rule)
if slow_learn:
    dot_brain_file = open("test_brain.txt", "rb")
    population.dots[0].dot_brain.read_from_stream(dot_brain_file)
    dot_brain_file.close()
    for i in range(1, 100):
        population.dots[i].dot_brain = population.dots[0].dot_brain.copy()
        population.dots[i].dot_brain.mutate(0.3, 0.5)
redraw(blank, fields, population)

cv2.destroyAllWindows()
