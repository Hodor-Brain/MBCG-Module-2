import json
import cv2
import numpy as np
import os.path

initial_points = None
a_control_points = None
b_control_points = None


def fit(points):
    global initial_points, a_control_points, b_control_points
    initial_points = np.array(points, dtype=np.float32)
    n = len(initial_points) - 1

    c_matrix = 4 * np.identity(n)
    np.fill_diagonal(c_matrix[1:], 1)
    np.fill_diagonal(c_matrix[:, 1:], 1)
    c_matrix[0, 0] = 2
    c_matrix[n - 1, n - 1] = 7
    c_matrix[n - 1, n - 2] = 2

    p_vector = [2 * (2 * initial_points[i] + initial_points[i + 1]) for i in range(n)]
    p_vector[0] = initial_points[0] + 2 * initial_points[1]
    p_vector[n - 1] = 8 * initial_points[n - 1] + initial_points[n]

    a_control_points = np.linalg.solve(c_matrix, p_vector)
    b_control_points = [0] * n
    for i in range(n - 1):
        b_control_points[i] = 2 * initial_points[i + 1] - a_control_points[i + 1]

    b_control_points[n - 1] = (a_control_points[n - 1] + initial_points[n]) / 2


def fit_point_to_curve(p1, a, b, p2, t):
    return ((1 - t) ** 3) * p1 + 3 * pow(1 - t, 2) * t * a + 3 * (1 - t) * (t ** 2) * b + (t ** 3) * p2


def draw(img):
    for i in range(0, len(initial_points) - 1):
        current_point = initial_points[i]
        next_point = initial_points[i + 1]
        a_control_point = a_control_points[i]
        b_control_point = b_control_points[i]

        t = np.linspace(0, 1, 50)
        for k in range(0, len(t) - 1):
            first = fit_point_to_curve(current_point, a_control_point, b_control_point, next_point, t[k])
            last = fit_point_to_curve(current_point, a_control_point, b_control_point, next_point, t[k + 1])

            cv2.line(img, (int(first[0]), int(first[1])), (int(last[0]), int(last[1])), (0, 255, 0), 3)

        cv2.circle(img, (int(current_point[0]), int(current_point[1])), 6, (255, 0, 0), -1)
        cv2.circle(img, (int(next_point[0]), int(next_point[1])), 6, (255, 0, 0), -1)


if __name__ == '__main__':
    file = open(os.path.join(os.getcwd(), '3.json'))
    points = json.load(file)['curve']

    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    fit(points)
    draw(img)
    cv2.imshow("Bezier Spline", img)
    cv2.waitKey(0)
