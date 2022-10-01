from classes import VertexRecord, HalfEdgeRecord, \
    FaceRecord, DoubleLinkedEdgeList
from map_overlay_algorithm import bentley_ottman_algorithm
from faces_fix import Graph
from visualizer import *

def dfs(half_edge, visited, incident_face, F):
    visited.add(half_edge)
    half_edge.incidental_face = incident_face
    F[incident_face].append(half_edge)

    if half_edge.next_edge not in visited:
        dfs(half_edge.next_edge, visited, incident_face, F)


def convert_edge_list_to_DCEL(edge_list):
    V = {}  # tuple (point) : VertexRecord

    for a, b in edge_list:
        if a not in V.keys():
            x, y = a
            V[a] = VertexRecord(x, y)
        if b not in V.keys():
            x, y = b
            V[b] = VertexRecord(x, y)

    E = {}  # VertexRecord : [HalfEdgeRecord]

    for a, b in edge_list:
        a_vertex = V[a]
        a_beginning_half_edge = HalfEdgeRecord(beginning=a_vertex)

        b_vertex = V[b]
        b_beginning_half_edge = HalfEdgeRecord(beginning=b_vertex)

        a_beginning_half_edge.twin_edge = b_beginning_half_edge
        b_beginning_half_edge.twin_edge = a_beginning_half_edge

        if a_vertex in E.keys():
            E[a_vertex].append(a_beginning_half_edge)
        else:
            E[a_vertex] = [a_beginning_half_edge]

        if b_vertex in E.keys():
            E[b_vertex].append(b_beginning_half_edge)
        else:
            E[b_vertex] = [b_beginning_half_edge]

    for v in E.keys():
        E[v].sort(reverse=True)

    for v in E.keys():
        v.incidental_edge = E[v][-1]
        half_edges = E[v]
        for edge_1, edge_2 in zip(half_edges, half_edges[1:] + [half_edges[0]]):
            edge_1.twin_edge.next_edge = edge_2
            edge_2.previous_edge = edge_1.twin_edge

    visited = set()
    F = {}  # FaceRecord : [HalfEdgeRecord]
    faces = []

    for v in E.keys():
        for half_edge in E[v]:
            if half_edge not in visited:
                i = len(faces)
                new_face = FaceRecord(i, outer_edge=half_edge)
                faces.append(new_face)
                F[new_face] = []
                dfs(half_edge, visited, new_face, F)

    edges = {}
    for v in V.values():
        for e in E[v]:
            if tuple(e) not in edges.keys() \
                    and tuple(e.twin_edge) not in edges.keys():
                edges[tuple(e)] = e

    vertices = V

    return DoubleLinkedEdgeList(vertices, edges, faces)



def map_input():
    plot = Plot()
    plot.scenes = []
    plot.add_scene(Scene(points=[PointsCollection([(0, 0), (100, 100)])],
                         title="To draw the first map click Add map.\n"
                               + "To draw the second map once again click Add map."))
    plot.draw()
    maps = plot.get_added_maps()
    plot.scenes = []

    if len(maps) == 0:
        raise Exception("No maps have been drawn")
    elif len(maps) == 1:
        maps = [maps[0], []]

    D1 = convert_edge_list_to_DCEL(maps[0])
    D2 = convert_edge_list_to_DCEL(maps[1])

    new_D, points_of_intersection = bentley_ottman_algorithm(D1, D2, 1e-05)

    plot.add_scene(Scene(points=[PointsCollection(points_of_intersection, color='red')],
                         lines=[LinesCollection(maps[0], color='black'),
                                LinesCollection(maps[1], color='blue')],
                         title=f"{len(points_of_intersection)} intersection points"))

    G = Graph(new_D.precision)
    G.initGraph(new_D)

    lines = [[] for _ in range(G.V)]

    for edge in new_D.edges.values():
        offset1 = calcOffset(edge)
        offset2 = calcOffset(edge.twin_edge)
        scale = 1
        lines[edge.G_vertex_id].append([[edge.beginning.x + scale*offset1[0],
                                         edge.beginning.y + scale*offset1[1]],
                                        [edge.twin_edge.beginning.x + scale*offset1[0],
                                         edge.twin_edge.beginning.y + scale*offset1[1]]])
        lines[edge.twin_edge.G_vertex_id].append([[edge.twin_edge.beginning.x + scale*offset2[0],
                                                   edge.twin_edge.beginning.y + scale * offset2[1]],
                                                  [edge.beginning.x + scale*offset2[0],
                                                   edge.beginning.y + scale*offset2[1]]])

    cmap = get_cmap(len(lines))

    linesCol = []
    for i in range(len(lines)):
        linesCol.append(LinesCollection(lines[i], color=cmap(i)))
    plot.add_scene(
        Scene(lines=linesCol, title="Faces with colored half edges"))

    faces = find_polygons(new_D)

    for lines in faces:
        plot.add_scene(Scene(lines=[LinesCollection(maps[0] + maps[1], color='blue'),
                                    LinesCollection(lines, color='red')],
                             title="Polygons in order of decreasing area size"))
    plot.draw()

    show_faces_as_polygons(faces)



if __name__ == "__main__":
    map_input()