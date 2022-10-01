from classes import DoubleLinkedEdgeList, \
    VertexRecord, HalfEdgeRecord, FaceRecord
from map_overlay_algorithm import bentley_ottman_algorithm, \
    AVLTree, makeGraph, fixFacesRecords
from visualizer import *
from faces_fix import *
import numpy as np
import matplotlib.pyplot as plt
import sys


def createTestDoubleLinkedEdgeLists():
    D1 = DoubleLinkedEdgeList()

    max_x = 0
    max_y = 0

    for _ in range(10):
        vertices = []
        # wkazywanie krawędzi - (następna, poprzednia)
        edges = []
        starting_vertex = VertexRecord(np.random.uniform(max_x, max_x+5),
                                       np.random.uniform(max_y, max_y+5))
        curr_vertex = starting_vertex

        delta_fi = np.random.uniform(-0.2, np.pi/2)
        delta_dist = 1
        next_vertex = VertexRecord(curr_vertex.x + delta_dist * np.cos(delta_fi),
                                   curr_vertex.y + delta_dist * np.sin(delta_fi))

        max_x = max(max_x, next_vertex.x)
        max_y = max(max_y, next_vertex.y)

        start_half_edge_1 = HalfEdgeRecord(beginning=curr_vertex)
        start_half_edge_2 = HalfEdgeRecord(beginning=next_vertex)
        start_half_edge_1.twin_edge = start_half_edge_2
        start_half_edge_2.twin_edge = start_half_edge_1

        curr_vertex.incidental_edge = start_half_edge_1

        vertices.append(starting_vertex)
        edges.append((start_half_edge_1, start_half_edge_2))

        while True:
            delta_fi = delta_fi + np.random.uniform(-0.2, np.pi/2)
            delta_dist = 1
            next_vertex = VertexRecord(curr_vertex.x + delta_dist * np.sin(delta_fi),
                                       curr_vertex.y + delta_dist * np.cos(delta_fi))

            max_x = max(max_x, next_vertex.x)
            max_y = max(max_y, next_vertex.y)

            half_edge_1 = HalfEdgeRecord(beginning=curr_vertex)
            half_edge_2 = HalfEdgeRecord(beginning=next_vertex)
            half_edge_1.twin_edge = half_edge_2
            half_edge_2.twin_edge = half_edge_1

            if D1.getSlope(half_edge_1, curr_vertex)[0] == 4:
                break

            curr_vertex.incidental_edge = half_edge_1

            edges[-1][0].next_edge = half_edge_1
            half_edge_1.previous_edge = edges[-1][0]
            edges[-1][1].previous_edge = half_edge_2
            half_edge_2.next_edge = edges[-1][1]

            vertices.append(next_vertex)
            edges.append((half_edge_1, half_edge_2))

            curr_vertex = next_vertex

        curr_vertex = vertices[-1]

        half_edge_3 = HalfEdgeRecord(beginning=vertices[-1])
        half_edge_4 = HalfEdgeRecord(beginning=starting_vertex)
        half_edge_3.twin_edge = half_edge_4
        half_edge_4.twin_edge = half_edge_3

        edges[-1][0].next_edge = half_edge_3
        half_edge_3.previous_edge = edges[-1][0]
        edges[-1][1].previous_edge = half_edge_4
        half_edge_4.next_edge = edges[-1][1]

        start_half_edge_1.previous_edge = half_edge_3
        half_edge_3.next_edge = start_half_edge_1
        half_edge_4.previous_edge = start_half_edge_2
        start_half_edge_2.next_edge = half_edge_4

        curr_vertex.incidental_edge = half_edge_3

        edges.append((half_edge_3, half_edge_4))

        for vertex in vertices:
            D1.vertices[(vertex.x, vertex.y)] = vertex
        for edge in edges:
            D1.edges[((edge[0].beginning.x, edge[0].beginning.y),
                      (edge[1].beginning.x, edge[1].beginning.y))] = edge[0]

    return D1


def createTestDoubleLinkedEdgeList_v2(number_of_hulls):
    vertices = []
    max_x = 0
    max_y = 0

    for i in range(number_of_hulls):
        curr_vertices = []
        delta_fi = 0
        x = max_x
        y = max_y
        starting_point = (x, y)
        curr_vertices.append(starting_point)
        while True:
            delta_fi += np.random.uniform(-0.2, np.pi / 4)
            delta_dist = np.random.uniform(0.5, 2.0)

            x += delta_dist * np.cos(delta_fi)
            y += delta_dist * np.sin(delta_fi)

            max_x = max(max_x, x)
            max_y = max(max_y, y)

            next_vertex = (x, y)

            if (next_vertex[1] - curr_vertices[-1][1]) \
                    / (next_vertex[0] - curr_vertices[-1][0]) > 1 \
                    and (next_vertex[1] - curr_vertices[-1][1]) < 0:
                break

            curr_vertices.append(next_vertex)

        vertices.append(curr_vertices)

    new_D = DoubleLinkedEdgeList()
    for hull in vertices:
        new_D.joinList(createTestDoubleLinkedEdgeListsFromArray(hull))

    return new_D


def createTestDoubleLinkedEdgeListsFromArray(arr):
    new_D1 = DoubleLinkedEdgeList()
    n = len(arr)

    vertices = []
    edges = []

    start_point1 = VertexRecord(arr[0][0], arr[0][1])
    start_point2 = VertexRecord(arr[1][0], arr[1][1])

    start_half_edge_1 = HalfEdgeRecord(beginning=start_point1)
    start_half_edge_2 = HalfEdgeRecord(beginning=start_point2)
    start_half_edge_1.twin_edge = start_half_edge_2
    start_half_edge_2.twin_edge = start_half_edge_1

    start_point1.incidental_edge = start_half_edge_1

    vertices.append(start_point1)
    vertices.append(start_point2)
    edges.append((start_half_edge_1, start_half_edge_2))

    for i in range(2, n):
        point1 = vertices[-1]
        point2 = VertexRecord(arr[i][0], arr[i][1])

        half_edge_1 = HalfEdgeRecord(beginning=point1)
        half_edge_2 = HalfEdgeRecord(beginning=point2)
        half_edge_1.twin_edge = half_edge_2
        half_edge_2.twin_edge = half_edge_1

        edges[-1][0].next_edge = half_edge_1
        half_edge_1.previous_edge = edges[-1][0]
        edges[-1][1].previous_edge = half_edge_2
        half_edge_2.next_edge = edges[-1][1]

        point1.incidental_edge = half_edge_1

        vertices.append(point2)
        edges.append((half_edge_1, half_edge_2))

    end_half_edge_1 = HalfEdgeRecord(beginning=vertices[-1])
    end_half_edge_2 = HalfEdgeRecord(beginning=vertices[0])

    end_half_edge_1.twin_edge = end_half_edge_2
    end_half_edge_2.twin_edge = end_half_edge_1

    end_half_edge_1.previous_edge = edges[-1][0]
    edges[-1][0].next_edge = end_half_edge_1
    end_half_edge_2.next_edge = edges[-1][1]
    edges[-1][1].previous_edge = end_half_edge_2

    end_half_edge_1.next_edge = start_half_edge_1
    start_half_edge_1.previous_edge = end_half_edge_1
    end_half_edge_2.previous_edge = start_half_edge_2
    start_half_edge_2.next_edge = end_half_edge_2

    vertices[-1].incidental_edge = end_half_edge_1

    edges.append((end_half_edge_1, end_half_edge_2))

    for vertex in vertices:
        new_D1.vertices[(vertex.x, vertex.y)] = vertex
    for edge in edges:
        new_D1.edges[((edge[0].beginning.x, edge[0].beginning.y),
                      (edge[1].beginning.x, edge[1].beginning.y))] = edge[0]

    return new_D1


def fillCycleWithFace(half_edge, face):
    # Cykl wewnętrzny
    half_edge.incidental_faces = [face]
    curr_edge = half_edge.next_edge
    while curr_edge != half_edge:
        curr_edge.incidental_faces = [face]
        curr_edge = curr_edge.next_edge

    # Cykle zewnętrzne dla wszystkich dziur
    for edge in face.inner_edges:
        curr_edge = edge
        curr_edge.incidental_faces = [face]
        curr_edge = curr_edge.next_edge
        while curr_edge != edge:
            curr_edge.incidental_faces = [face]
            curr_edge = curr_edge.next_edge


def test_collinear(j):
    plot = Plot()
    plot.scenes = []
    if j == 0:
        # test 10.1 - nachodzące na siebie krawędzie (jedna zawiera się w drugiej)
        D1 = createTestDoubleLinkedEdgeListsFromArray([(0, 0), (50, 10), (40, 30), (20, 30), (0, 20), (-20, 20)])
        D2 = createTestDoubleLinkedEdgeListsFromArray([(-5, 5), (10, 10), (-10, 10)])

    if j == 1:
        # test 10.2 - wejściwe krawędzie nachodzące na siebie i kończące w jednym wierzchołku
        D1 = createTestDoubleLinkedEdgeListsFromArray([(0, 0), (50, 10), (40, 30), (20, 30), (0, 20), (-20, 20)])
        D2 = createTestDoubleLinkedEdgeListsFromArray([(0, 0), (10, 10), (-10, 10)])

    if j == 2:
        # test 10.3 - wejściowe 2 takie same krawędzie (całkowicie zawierające się w sobie)
        D1 = createTestDoubleLinkedEdgeListsFromArray([(30, 10), (90, 10), (90, 50), (10, 50), (10, 30)])
        D2 = createTestDoubleLinkedEdgeListsFromArray([(30, 10), (50, 30), (30, 50), (10, 30)])

    edges_1 = [((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))
               for e in D1.edges.values()]
    edges_2 = [((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))
               for e in D2.edges.values()]
    plot.add_scene(Scene(lines=[LinesCollection(edges_1, color='blue'),
                                LinesCollection(edges_2, color='black')],
                         title=f"Collinear edges test {j+1}"))

    new_D, points_of_intersection = bentley_ottman_algorithm(D1, D2, 1e-05)

    plot.add_scene(Scene(points=[PointsCollection(points=points_of_intersection, color='red')], 
                         lines=[LinesCollection(lines=[((e.beginning.x, e.beginning.y), 
                                                       (e.twin_edge.beginning.x, e.twin_edge.beginning.y)) 
                                                     for e in new_D.edges.values()], color='black')],
                         title=f'{len(points_of_intersection)} intersection points'))

    # Inicjalizacja grafu dla naprawiania rekordów ścian
    helper_A = AVLTree(new_D.precision)
    G = Graph(new_D.precision)
    G.initGraph(new_D)

    makeGraph(new_D, G, new_D.precision)

    lines = [[] for _ in range(G.V)]

    for edge in new_D.edges.values():
        offset1 = calcOffset(edge)
        offset2 = calcOffset(edge.twin_edge)
        lines[edge.G_vertex_id].append([[edge.beginning.x + offset1[0], edge.beginning.y + offset1[1]], [edge.twin_edge.beginning.x + offset1[0], edge.twin_edge.beginning.y + offset1[1]]])
        lines[edge.twin_edge.G_vertex_id].append([[edge.twin_edge.beginning.x + offset2[0], edge.twin_edge.beginning.y + offset2[1]], [edge.beginning.x + offset2[0], edge.beginning.y + offset2[1]]])

    cmap = get_cmap(len(lines))

    linesCol = []
    for i in range(len(lines)):
        linesCol.append(LinesCollection(lines[i], color=cmap(i)))
    plot.add_scene(Scene(lines=linesCol, title="Faces with colored half edges"))

    faces = find_polygons(new_D)

    for lines in faces:
        plot.add_scene(Scene(lines=[LinesCollection(edges_1 + edges_2, color='blue'),
                                    LinesCollection(lines, color='red')],
                             title="Polygons in order of decreasing area size"))

    plot.draw()


def test_random():
    plot = Plot()
    plot.scenes = []
    D1 = createTestDoubleLinkedEdgeList_v2(3)
    D2 = createTestDoubleLinkedEdgeList_v2(4)
    edges_1 = [((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))
               for e in D1.edges.values()]
    edges_2 = [((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))
               for e in D2.edges.values()]
    plot.add_scene(Scene(lines=[LinesCollection(edges_1, color='blue'),
                                LinesCollection(edges_2, color='black')],
                         title="Randomly generated maps"))

    new_D, points_of_intersection = bentley_ottman_algorithm(D1, D2, 1e-05)
    edges_3 = [((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))
               for e in new_D.edges.values()]

    plot.add_scene(Scene(points=[PointsCollection(points_of_intersection, color='red')],
                         lines=[LinesCollection(edges_3, color='black')],
                         title=f"{len(points_of_intersection)} intersection points"))

    G = Graph(new_D.precision)
    G.initGraph(new_D)

    lines = [[] for _ in range(G.V)]

    for edge in new_D.edges.values():
        offset1 = calcOffset(edge)
        offset2 = calcOffset(edge.twin_edge)
        lines[edge.G_vertex_id].append([[edge.beginning.x + offset1[0],
                                         edge.beginning.y + offset1[1]],
                                        [edge.twin_edge.beginning.x + offset1[0],
                                         edge.twin_edge.beginning.y + offset1[1]]])
        lines[edge.twin_edge.G_vertex_id].append([[edge.twin_edge.beginning.x + offset2[0],
                                                   edge.twin_edge.beginning.y + offset2[1]],
                                                  [edge.beginning.x + offset2[0],
                                                   edge.beginning.y + offset2[1]]])

    cmap = get_cmap(len(lines))
    linesCol = []
    for i in range(len(lines)):
        linesCol.append(LinesCollection(lines[i], color=cmap(i)))
    plot.add_scene(
        Scene(lines=linesCol, title="Faces with colored half edges"))

    faces = find_polygons(new_D)

    for lines in faces:
        plot.add_scene(Scene(lines=[LinesCollection(edges_1 + edges_2, color='blue'),
                                    LinesCollection(lines, color='red')],
                             title="Polygons in order of decreasing area size"))

    plot.draw()


def test_faces_holes_vertex_merge():
    plot = Plot()
    plot.scenes = []
    # test 10.4 - połączenia pomiędzy dziurami i cyklami
    D1 = DoubleLinkedEdgeList()
    D2 = DoubleLinkedEdgeList()

    D1.joinList(createTestDoubleLinkedEdgeListsFromArray([(10, 10), (100, 10), (100, 80), (10, 80)]))
    D1.joinList(createTestDoubleLinkedEdgeListsFromArray([(50, 60), (70, 50), (90, 60), (70, 70)]))

    face1 = FaceRecord(1, D1.getEdgeFromTuple(((10, 10), (100, 10))), [D1.getEdgeFromTuple(((70, 50), (50, 60)))], D1.getEdgeFromTuple(((100, 10), (10, 10))))
    face1.data = 50
    face2 = FaceRecord(2, D1.getEdgeFromTuple(((70, 50), (90, 60))), None, D1.getEdgeFromTuple(((70, 50), (50, 60))))
    face2.data = 70

    fillCycleWithFace(D1.getEdgeFromTuple(((10, 10), (100, 10))), face1)
    fillCycleWithFace(D1.getEdgeFromTuple(((70, 50), (90, 60))), face2)

    D2.joinList(createTestDoubleLinkedEdgeListsFromArray([(20, 60), (40, 50), (50, 60), (40, 70)]))
    D2.joinList(createTestDoubleLinkedEdgeListsFromArray([(30, 30), (40, 20), (50, 30), (40, 40)]))
    D2.joinList(createTestDoubleLinkedEdgeListsFromArray([(60, 30), (70, 20), (80, 30), (70, 40)]))

    face3 = FaceRecord(1, D2.getEdgeFromTuple(((20, 60), (40, 50))), None, D2.getEdgeFromTuple(((40, 50), (20, 60))))
    face3.data = 20
    face4 = FaceRecord(2, D2.getEdgeFromTuple(((30, 30), (40, 20))), None, D2.getEdgeFromTuple(((40, 20), (30, 30))))
    face4.data = 30
    face5 = FaceRecord(3, D2.getEdgeFromTuple(((60, 30), (70, 20))), None, D2.getEdgeFromTuple(((70, 20), (60, 30))))
    face5.data = 50

    fillCycleWithFace(D2.getEdgeFromTuple(((20, 60), (40, 50))), face3)
    fillCycleWithFace(D2.getEdgeFromTuple(((30, 30), (40, 20))), face4)
    fillCycleWithFace(D2.getEdgeFromTuple(((60, 30), (70, 20))), face5)

    D1.faces += [face1, face2]
    D2.faces += [face3, face4, face5]

    edges_1 = [((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))
               for e in D1.edges.values()]
    edges_2 = [((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))
               for e in D2.edges.values()]
    plot.add_scene(Scene(lines=[LinesCollection(edges_1, color='blue'),
                                LinesCollection(edges_2, color='black')],
                         title="Faces/holes/vertices merging test"))

    new_D, points_of_intersection = bentley_ottman_algorithm(D1, D2, 1e-05)

    plot.add_scene(Scene(points=[PointsCollection(points=points_of_intersection, color='red')], 
                         lines=[LinesCollection(lines=[((e.beginning.x, e.beginning.y), 
                                                        (e.twin_edge.beginning.x, e.twin_edge.beginning.y)) 
                                                       for e in new_D.edges.values()], color='black')],
                         title=f"{len(points_of_intersection)} intersection points"))

    # Inicjalizacja grafu dla naprawiania rekordów ścian
    helper_A = AVLTree(new_D.precision)
    G = Graph(new_D.precision)
    G.initGraph(new_D)
    makeGraph(new_D, G, new_D.precision)
    fixFacesRecords(new_D, G)
    lines = [[] for _ in range(G.V)]

    for edge in new_D.edges.values():
        offset1 = calcOffset(edge)
        offset2 = calcOffset(edge.twin_edge)
        lines[edge.G_vertex_id].append([[edge.beginning.x + offset1[0], edge.beginning.y + offset1[1]],
                                        [edge.twin_edge.beginning.x + offset1[0],
                                         edge.twin_edge.beginning.y + offset1[1]]])
        lines[edge.twin_edge.G_vertex_id].append(
            [[edge.twin_edge.beginning.x + offset2[0], edge.twin_edge.beginning.y + offset2[1]],
             [edge.beginning.x + offset2[0], edge.beginning.y + offset2[1]]])
    cmap = get_cmap(len(lines))

    # for face in new_D.faces:
    #     print(face.face_id, ":", face.data)

    linesCol = []
    # colors = ['blue', 'red', 'green', 'yellow', 'black', 'grey', 'purple']
    for i in range(len(lines)):
        linesCol.append(LinesCollection(lines[i], color=cmap(i)))
    plot.add_scene(Scene(lines=linesCol, title="Faces with colored half edges"))
    
    plot.add_scene(visualize(new_D, G))


    faces = find_polygons(new_D)
    for lines in faces:
        plot.add_scene(Scene(lines=[LinesCollection(edges_1 + edges_2, color='blue'),
                                    LinesCollection(lines, color='red')],
                             title="Polygons in order of decreasing area size"))

    plot.draw()


def test_main_test_with_faces():
    plot = Plot()
    plot.scenes = []
    D1 = DoubleLinkedEdgeList()
    D2 = DoubleLinkedEdgeList()

    D1.joinList(createTestDoubleLinkedEdgeListsFromArray([(50, 80), (70, 80), (90, 100), (70, 120), (50, 120), (30, 100)]))
    D1.joinList(createTestDoubleLinkedEdgeListsFromArray([(30, 40), (40, 30), (60, 30), (80, 40), (60, 50), (40, 50)]))
    D1.joinList(createTestDoubleLinkedEdgeListsFromArray([(110, 90), (130, 90), (130, 120), (110, 120)]))
    D1.joinList(createTestDoubleLinkedEdgeListsFromArray([(140, 90), (160, 90), (160, 120), (140, 120)]))
    D1.joinList(createTestDoubleLinkedEdgeListsFromArray([(170, 90), (190, 90), (190, 120), (170, 120)]))
    D1.joinList(createTestDoubleLinkedEdgeListsFromArray([(110, 30), (130, 30), (130, 50), (110, 50)]))

    face1 = FaceRecord(1, D1.getEdgeFromTuple(((50, 80), (70, 80))), None, D1.getEdgeFromTuple(((70, 80), (50, 80))))
    face1.data = 50
    face2 = FaceRecord(2, D1.getEdgeFromTuple(((30, 40), (40, 30))), None, D1.getEdgeFromTuple(((40, 30), (30, 40))))
    face2.data = 70
    face3 = FaceRecord(3, D1.getEdgeFromTuple(((110, 90), (130, 90))), None, D1.getEdgeFromTuple(((130, 90), (110, 90))))
    face3.data = 40
    face4 = FaceRecord(4, D1.getEdgeFromTuple(((140, 90), (160, 90))), None, D1.getEdgeFromTuple(((160, 90), (140, 90))))
    face4.data = 30
    face5 = FaceRecord(5, D1.getEdgeFromTuple(((170, 90), (190, 90))), None, D1.getEdgeFromTuple(((190, 90), (170, 90))))
    face5.data = 60
    face6 = FaceRecord(6, D1.getEdgeFromTuple(((110, 30), (130, 30))), None, D1.getEdgeFromTuple(((130, 30), (110, 30))))
    face6.data = 30

    fillCycleWithFace(D1.getEdgeFromTuple(((50, 80), (70, 80))), face1)
    fillCycleWithFace(D1.getEdgeFromTuple(((30, 40), (40, 30))), face2)
    fillCycleWithFace(D1.getEdgeFromTuple(((110, 90), (130, 90))), face3)
    fillCycleWithFace(D1.getEdgeFromTuple(((140, 90), (160, 90))), face4)
    fillCycleWithFace(D1.getEdgeFromTuple(((170, 90), (190, 90))), face5)
    fillCycleWithFace(D1.getEdgeFromTuple(((110, 30), (130, 30))), face6)

    D2.joinList(createTestDoubleLinkedEdgeListsFromArray([(10, 100), (30, 80), (50, 80), (70, 100), (50, 102), (30, 102)]))
    D2.joinList(createTestDoubleLinkedEdgeListsFromArray([(20, 10), (80, 10), (80, 50), (70, 70), (30, 70), (10, 50)]))
    D2.joinList(createTestDoubleLinkedEdgeListsFromArray([(60, 30), (70, 40), (60, 50)]))
    D2.joinList(createTestDoubleLinkedEdgeListsFromArray([(100, 10), (210, 10), (210, 130), (100, 130)]))

    face7 = FaceRecord(1, D2.getEdgeFromTuple(((10, 100), (30, 80))), None, D2.getEdgeFromTuple(((30, 80), (10, 100))))
    face7.data = 60
    face8 = FaceRecord(2, D2.getEdgeFromTuple(((20, 10), (80, 10))), [D2.getEdgeFromTuple(((60, 30), (70, 40)))], D2.getEdgeFromTuple(((80, 10), (20, 10))))
    face8.data = 70
    face9 = FaceRecord(3, D2.getEdgeFromTuple(((60, 30), (70, 40))), None, D2.getEdgeFromTuple(((70, 40), (60, 30))))
    face9.data = 50
    face10 = FaceRecord(4, D2.getEdgeFromTuple(((100, 10), (210, 10))), None, D2.getEdgeFromTuple(((210, 10), (100, 10))))
    face10.data = 50

    fillCycleWithFace(D2.getEdgeFromTuple(((10, 100), (30, 80))), face7)
    fillCycleWithFace(D2.getEdgeFromTuple(((20, 10), (80, 10))), face8)
    fillCycleWithFace(D2.getEdgeFromTuple(((60, 30), (70, 40))), face9)
    fillCycleWithFace(D2.getEdgeFromTuple(((100, 10), (210, 10))), face10)

    D1.faces += [face1, face2, face3, face4, face5, face6]
    D2.faces += [face7, face8, face9, face10]

    edges_1 = [((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))
               for e in D1.edges.values()]
    edges_2 = [((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))
               for e in D2.edges.values()]
    plot.add_scene(Scene(lines=[LinesCollection(edges_1, color='blue'),
                                LinesCollection(edges_2, color='black')],
                         title="Faces/holes/vertices merging test"))

    new_D, points_of_intersection = bentley_ottman_algorithm(D1, D2, 1e-05)

    plot.add_scene(Scene(points=[PointsCollection(points=points_of_intersection, color='red')],
                         lines=[LinesCollection(lines=[((e.beginning.x, e.beginning.y),
                                                        (e.twin_edge.beginning.x, e.twin_edge.beginning.y))
                                                       for e in new_D.edges.values()], color='black')],
                         title=f"{len(points_of_intersection)} intersection points"))

    # Inicjalizacja grafu dla naprawiania rekordów ścian
    helper_A = AVLTree(new_D.precision)
    G = Graph(new_D.precision)
    G.initGraph(new_D)
    makeGraph(new_D, G, new_D.precision)
    fixFacesRecords(new_D, G)
    lines = [[] for _ in range(G.V)]

    for edge in new_D.edges.values():
        offset1 = calcOffset(edge)
        offset2 = calcOffset(edge.twin_edge)
        lines[edge.G_vertex_id].append([[edge.beginning.x + offset1[0], edge.beginning.y + offset1[1]],
                                        [edge.twin_edge.beginning.x + offset1[0],
                                         edge.twin_edge.beginning.y + offset1[1]]])
        lines[edge.twin_edge.G_vertex_id].append(
            [[edge.twin_edge.beginning.x + offset2[0], edge.twin_edge.beginning.y + offset2[1]],
             [edge.beginning.x + offset2[0], edge.beginning.y + offset2[1]]])
    cmap = get_cmap(len(lines))

    # for face in new_D.faces:
    #     print(face.face_id, ":", face.data)

    linesCol = []
    # colors = ['blue', 'red', 'green', 'yellow', 'black', 'grey', 'purple']
    for i in range(len(lines)):
        linesCol.append(LinesCollection(lines[i], color=cmap(i)))
    plot.add_scene(Scene(lines=linesCol, title="Faces with colored half edges"))

    plot.add_scene(visualize(new_D, G))

    faces = find_polygons(new_D)
    for lines in faces:
        plot.add_scene(Scene(lines=[LinesCollection(edges_1 + edges_2, color='blue'),
                                    LinesCollection(lines, color='red')],
                             title="Polygons in order of decreasing area size"))

    plot.draw()


if __name__ == '__main__':
    sample_D = DoubleLinkedEdgeList()

    test_main_test_with_faces()

    test_collinear(0)
    test_collinear(1)
    test_collinear(2)

    test_faces_holes_vertex_merge()

    for _ in range(5):
        test_random()

