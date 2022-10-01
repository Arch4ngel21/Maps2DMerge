import numpy as np
from classes import HalfEdgeRecord, VertexRecord

class Graph:
    def __init__(self, precision):
        self.V = 0
        self.adjList = []
        self.lowestLeftVertices = []
        self.vertexType = []
        self.edgeInCycle = []
        self.precision = precision

        # key - (x, y), value - G_index
        self.representativesLookup = {}

    def initGraph(self, D):
        round_place = int(-np.log10(self.precision))
        check_edges = {}
        for edge in D.edges.values():
            check_edges[((round(edge.beginning.x, round_place),
                          round(edge.beginning.y, round_place)),
                         (round(edge.twin_edge.beginning.x, round_place),
                          round(edge.twin_edge.beginning.y, round_place)))] = False
            check_edges[((round(edge.twin_edge.beginning.x, round_place),
                          round(edge.twin_edge.beginning.y, round_place)),
                         (round(edge.beginning.x, round_place),
                          round(edge.beginning.y, round_place)))] = False

        for edge in D.edges.values():
            if not check_edges[((round(edge.beginning.x, round_place),
                                 round(edge.beginning.y, round_place)),
                               (round(edge.twin_edge.beginning.x, round_place),
                                round(edge.twin_edge.beginning.y, round_place)))]:
                check_edges[((round(edge.beginning.x, round_place),
                              round(edge.beginning.y, round_place)),
                             (round(edge.twin_edge.beginning.x, round_place),
                              round(edge.twin_edge.beginning.y, round_place)))] = True
                self.adjList.append([])
                self.vertexType.append(-1)
                self.lowestLeftVertices.append(edge.beginning)
                self.V += 1
                self.edgeInCycle.append(edge)

                edge.G_vertex_id = self.V - 1

                curr_edge = edge.next_edge
                while not check_edges[((round(curr_edge.beginning.x, round_place),
                                        round(curr_edge.beginning.y, round_place)),
                                       (round(curr_edge.twin_edge.beginning.x, round_place),
                                        round(curr_edge.twin_edge.beginning.y, round_place)))]:
                    check_edges[((round(curr_edge.beginning.x, round_place),
                                  round(curr_edge.beginning.y, round_place)),
                                 (round(curr_edge.twin_edge.beginning.x, round_place),
                                  round(curr_edge.twin_edge.beginning.y, round_place)))] = True
                    curr_edge.G_vertex_id = self.V - 1
                    if self.isLowerVertex(self.lowestLeftVertices[self.V - 1], curr_edge.beginning):
                        self.lowestLeftVertices[self.V - 1] = curr_edge.beginning
                    curr_edge = curr_edge.next_edge
                self.findVertexType(curr_edge)

            if not check_edges[((round(edge.twin_edge.beginning.x, round_place),
                                 round(edge.twin_edge.beginning.y, round_place)),
                               (round(edge.beginning.x, round_place),
                                round(edge.beginning.y, round_place)))]:
                check_edges[((round(edge.twin_edge.beginning.x, round_place),
                              round(edge.twin_edge.beginning.y, round_place)),
                             (round(edge.beginning.x, round_place),
                              round(edge.beginning.y, round_place)))] = True
                self.adjList.append([])
                self.vertexType.append(-1)
                self.lowestLeftVertices.append(edge.twin_edge.beginning)
                self.V += 1
                self.edgeInCycle.append(edge.twin_edge)

                edge.twin_edge.G_vertex_id = self.V - 1

                curr_edge = edge.twin_edge.next_edge
                while not check_edges[((round(curr_edge.beginning.x, round_place),
                                        round(curr_edge.beginning.y, round_place)),
                                       (round(curr_edge.twin_edge.beginning.x, round_place),
                                        round(curr_edge.twin_edge.beginning.y, round_place)))]:
                    check_edges[((round(curr_edge.beginning.x, round_place),
                                  round(curr_edge.beginning.y, round_place)),
                                 (round(curr_edge.twin_edge.beginning.x, round_place),
                                  round(curr_edge.twin_edge.beginning.y, round_place)))] = True
                    curr_edge.G_vertex_id = self.V - 1
                    if self.isLowerVertex(self.lowestLeftVertices[self.V - 1], curr_edge.beginning):
                        self.lowestLeftVertices[self.V - 1] = curr_edge.beginning
                    curr_edge = curr_edge.next_edge
                self.findVertexType(curr_edge)

        for e, vertex in enumerate(self.lowestLeftVertices):
            # Dodajemy tylko reprezentantów dla cykli zewnętrznych
            if self.vertexType[e]:
                self.representativesLookup[(vertex.x, vertex.y)] = e

    def findVertexType(self, starting_edge):
        curr_edge = starting_edge
        while not self.isSamePoint(self.lowestLeftVertices[curr_edge.next_edge.G_vertex_id], curr_edge.next_edge.beginning):
            curr_edge = curr_edge.next_edge

        side = self.det2x2(curr_edge.beginning, curr_edge.next_edge.next_edge.beginning, curr_edge.next_edge.beginning)
        if side == 1:
            self.vertexType[curr_edge.next_edge.G_vertex_id] = 1
        elif side == 2:
            self.vertexType[curr_edge.next_edge.G_vertex_id] = 0
        else:
            print("Blad przy znajdowaniu wierzcholka dla cyklu -> det 3 punktow wynosi 0")

    def fillCycleWithFace(self, face, vertex_id):
        # Ustawienie rekordu ściany dla cyklu ograniczającego ścianę
        starting_edge = self.edgeInCycle[vertex_id]
        starting_edge.incidental_faces = [face]
        curr_edge = starting_edge.next_edge

        while curr_edge != starting_edge:
            starting_edge.incidental_faces = [face]
            curr_edge = curr_edge.next_edge

        # Ustawienie rekordu ściany dla zewnętrznych cykli dziur
        for i in self.adjList[vertex_id]:
            if self.vertexType[i]:
                starting_edge = self.edgeInCycle[i]
                starting_edge.incidental_faces = [face]
                curr_edge = starting_edge.next_edge

                while curr_edge != starting_edge:
                    starting_edge.incidental_faces = [face]
                    curr_edge = curr_edge.next_edge

    def getDataFromOuterCycle(self, vertex_id, DCEL_id):

        outer_vertex_id = vertex_id
        found_outer_cycle = True

        while found_outer_cycle:
            found_outer_cycle = False

            for i in self.adjList[outer_vertex_id]:
                # Znaleziono zewnętrzny cykl (tzn taki, który zawiera w sobie ten, z którego szukamy)
                if i != -1 and self.vertexType[i] == 0:
                    outer_vertex_id = i
                    found_outer_cycle = True
                    if self.edgeInCycle[i].incidental_faces[0].data[DCEL_id-1] != None:
                        return self.edgeInCycle[i].incidental_faces[0].data[DCEL_id-1]
                    continue

        return None

    def DFSSearchGetDataFromOuterCycle(self, vertex_id, DCEL_id, visited):
        visited[vertex_id] = True

        for i in self.adjList[vertex_id]:
            if i != -1 and not visited[i]:
                if self.vertexType[i] == 0 and self.edgeInCycle[i].incidental_faces[0].data[DCEL_id-1] != None:
                    return self.edgeInCycle[i].incidental_faces[0].data[DCEL_id-1]

                # Cykl zewnętrzny (inna dziura)
                if self.vertexType[i]:
                    res = self.DFSSearchGetDataFromOuterCycle(i, DCEL_id, visited)
                else:
                    # Jeżeli jest wewnętrzny cykl, ale nie ma informacji, której potrzebujemy, to kontynuujemy szukanie
                    # z okalającego go zewnętrznego cyklu (szukanie na zewnątrz w celu znalezienia informacji)
                    res = self.DFSSearchGetDataFromOuterCycle(self.edgeInCycle[i].incidental_faces[0].edge_to_outer_cycle.G_vertex_id, DCEL_id, visited)
                if res != None:
                    return res

        visited[vertex_id] = False
        return None

    def getEdgeToOuterCycle(self, vertex_id):
        curr_edge = self.edgeInCycle[vertex_id]
        while not self.vertexType[curr_edge.twin_edge.G_vertex_id]:
            curr_edge = curr_edge.next_edge
            if curr_edge == self.edgeInCycle[vertex_id]:
                return HalfEdgeRecord(beginning=VertexRecord(None, None))

        return curr_edge.twin_edge

    def getEdgesArrToInnerCycles(self, vertex_id, edges_arr):
        for i in self.adjList[vertex_id]:
            if self.vertexType[i]:
                if not self.doesContainVal(edges_arr, i):
                    edges_arr[i] = True
                    self.getEdgesArrToInnerCycles(i, edges_arr)

    def det2x2(self, p0, p1, p2):
        matrix = np.array([[p0.x - p2.x, p1.x - p2.x],
                           [p0.y - p2.y, p1.y - p2.y]])

        det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        if abs(det) < self.precision:
            return 0
        elif det > 0:
            return 1
        else:
            return 2

    def doesContainVal(self, dictionary, val):
        try:
            dictionary[val]
            return True
        except KeyError:
            return False

    def doesContainFace(self, dictionary, face):
        try:
            dictionary[(face.face_id, face.DCEL_id)]
            return True
        except KeyError:
            return False

    def isInLookup(self, vertex):
        try:
            self.representativesLookup[(vertex.x, vertex.y)]
            return True
        except KeyError:
            return False

    def isLowerVertex(self, curr_vertex, new_vertex):
        if new_vertex.x < curr_vertex.x:
            return True

        elif self.isSameVal(new_vertex.x, curr_vertex.x) and new_vertex.y < curr_vertex.y:
            return True

        return False

    def isSameVal(self, val1, val2):
        return abs(val1 - val2) < self.precision

    def isSamePoint(self, point1, point2):
        return abs(point1.x - point2.x) < self.precision and abs(point1.y - point2.y) < self.precision


# Znajdywanie lewego sąsiedniego cyklu dla wierzchołka w drzewie AVL odbywać się będzie za pomocą kończących się w
# tym wierzchołku krawędzi o przypisanym odpowiednim indeksie wierzchołka w G. W takim wierzchołku mogą być max. 2
# takie krawędzie (to jest cykl, więc jedna wchodzi i jedna wychodzi).
