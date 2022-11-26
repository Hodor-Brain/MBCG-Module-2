import json
import math
import random
import open3d as o3d
import numpy as np
import os.path

initial_points = None
matrix_length = None
spline_degree = None
S_col = None
S_row = None
knots_col = None
knots_row = None
control_points = None
weights = None


def init_surface_vars(points, gridSize, indices, degree):
    global matrix_length, spline_degree, initial_points
    matrix_length = gridSize[0]
    spline_degree = degree
    initial_points = np.zeros((matrix_length, matrix_length, 3))

    for i, pos in enumerate(indices):
        initial_points[pos[0], pos[1]] = points[i]

    get_s()
    generate_knots()
    generate_control_points()


def get_s():
    global S_row, S_col
    column_array, row_array = get_parameters()

    S_col = calculate_average_s(column_array)
    S_row = calculate_average_s(row_array)


def get_parameters():
    n = matrix_length
    column_array, row_array = np.zeros((n, n), dtype=np.float32), np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        column, row = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        column[0], column[n - 1] = 0., 1.
        row[0], row[n - 1] = 0., 1.

        d_column = sum(get_distance(initial_points[:, i][j], initial_points[:, i][j - 1]) for j in range(1, n))
        d_row = sum(get_distance(initial_points[i][j], initial_points[i][j - 1]) for j in range(1, n))

        for j in range(1, n - 1):
            column[j] = column[j - 1] + get_distance(initial_points[:, i][j], initial_points[:, i][j - 1]) / d_column
            row[j] = row[j - 1] + get_distance(initial_points[i][j], initial_points[i][j - 1]) / d_row

        column_array[i], row_array[i] = column, row

    return column_array, row_array


def get_distance(point_a, point_b):
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2 +
                     (point_a[2] - point_b[2]) ** 2)


def calculate_average_s(parameters):
    n = matrix_length
    result = np.zeros(n, dtype=np.float32)
    for i in range(n):
        vector = parameters[:, i]
        result[i] = sum(vector[j] for j in range(n)) / n
    return result


def generate_knots():
    global knots_row, knots_col
    n, k = matrix_length, spline_degree
    knots_col, knots_row = np.zeros(n + k + 1, dtype=np.float32), np.zeros(n + k + 1, dtype=np.float32)

    for i in range(1, n - k):
        knots_col[i + k] = (1. / k) * sum(S_col[j] for j in range(i, i + k))
        knots_row[i + k] = (1. / k) * sum(S_row[j] for j in range(i, i + k))

    for i in range(k + 1):
        knots_col[i], knots_row[i] = 0., 0.
        knots_col[n + k - i], knots_row[n + k - i] = 1., 1.


def generate_control_points():
    global control_points
    n, k = matrix_length, spline_degree

    q, control_points = np.zeros((n, n, 3)), np.zeros((n, n, 3))
    basis_function_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            basis_function_matrix[i, j] = basis_function(S_col[i], j, k, knots_col)
    for i in range(n):
        q[:, i] = np.linalg.solve(basis_function_matrix, initial_points[:, i])

    for i in range(n):
        for j in range(n):
            basis_function_matrix[i, j] = basis_function(S_row[i], j, k, knots_row)
    for i in range(n):
        control_points[i] = np.linalg.solve(basis_function_matrix, q[i])


def basis_function(t, i, k, knot_vector):
    if k == 0:
        if t == knot_vector[i] or knot_vector[i] < t < knot_vector[i + 1] or \
                (t == knot_vector[len(knot_vector) - 1] and i == len(knot_vector) - spline_degree - 2):
            return 1
        return 0

    left_summand, right_summand = 0., 0.
    left_denom = (knot_vector[i + k] - knot_vector[i])
    right_denom = (knot_vector[i + k + 1] - knot_vector[i + 1])

    if left_denom != 0:
        left_summand = (t - knot_vector[i]) * basis_function(t, i, k - 1, knot_vector) / left_denom
    if right_denom != 0:
        right_summand = (knot_vector[i + k + 1] - t) * basis_function(t, i + 1, k - 1, knot_vector) / right_denom

    return left_summand + right_summand


def generate_weights(min_weight, max_weight):
    global weights
    n = matrix_length
    weights = np.ones((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            weights[i, j] = random.uniform(min_weight, max_weight)


def draw_surface(surface_length, points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    mesh = create_mesh(surface_length)
    o3d.visualization.draw_geometries([point_cloud, mesh], mesh_show_wireframe=True, mesh_show_back_face=True)


def create_mesh(surface_length):
    points_array = create_points(surface_length)
    triangle_indexes = triangulate(surface_length)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_array)
    mesh.triangles = o3d.utility.Vector3iVector(triangle_indexes)
    return mesh


def create_points(surface_length):
    points_on_surface = []
    u, v = np.linspace(0, 1, surface_length), np.linspace(0, 1, surface_length)

    for i in range(len(u)):
        for j in range(len(v)):
            points_on_surface.append(fit_point_to_surface(u[i], v[j]))
    return points_on_surface


def fit_point_to_surface(u, v):
    n, k = matrix_length, spline_degree
    point = np.zeros(3)
    denominator = 0.

    for i in range(n):
        temp_point = sum(basis_function(v, j, k, knots_row) * control_points[i, j] * weights[i, j] for j in range(n))
        point += temp_point * basis_function(u, i, k, knots_col)

        temp = sum(basis_function(v, j, k, knots_row) * weights[i, j] for j in range(n))
        denominator += temp * basis_function(u, i, k, knots_col)

    return point / denominator


def triangulate(n):
    triangles = []

    for i in range(n - 1):
        for j in range(n - 1):
            triangles.append(np.array([i * n + j, i * n + j + 1, (i + 1) * n + j]).astype(np.int32))
            triangles.append(np.array([i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1]).astype(np.int32))

    return np.asarray(triangles)


if __name__ == '__main__':
    file = open(os.path.join(os.getcwd(), '3.json'))
    surface_data = json.load(file)["surface"]
    points, indices, gridSize = surface_data["points"], surface_data["indices"], surface_data["gridSize"]

    min_weight, max_weight, surface_length, degree = 0, 1, 25, 3

    init_surface_vars(points, gridSize, indices, degree)
    generate_weights(min_weight, max_weight)
    draw_surface(surface_length, points)
