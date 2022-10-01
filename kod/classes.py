import sys
import numpy as np

"""
QueueRecord - struktura danych dla przechowywania odpowiednich informacji w Kolejce Zdarzeń.
    x, y - wspołrzędne punktu, który reprezentuje dany rekord. Według nich będzie 
            również ustalany porządek w drzewie.
    edges - tablica wszystkich krawędzi odwiedzających wierzchołek
    left, right - odpowiednio prawe i lewe dziecko wierzchołka
    height - wysokość drzewa w tym wierzchołku
    type - typ rekordu: 0 - początek odcinka, 1 - koniec odcinka, 2 - przecięcie odcinków
    U - tablica odcinków, dla których p jest górnym końcem
    L - tablica odcinków, dla których p jest dolnym końcem
    C - tablica odcinków, które zawierają p w swoim wnętrzu
"""


class QueueRecord:
    def __init__(self, x, y, record_type, edges_index=None):
        self.x = x
        self.y = y

        self.left = None
        self.right = None
        self.parent = None
        self.height = 1
        self.U = []
        self.L = {}
        self.C = {}

        if record_type == 0:
            self.U = [edges_index]
        elif record_type == 1:
            self.L[edges_index] = True
        else:
            if not doesContainEdgeIndex(self.C, edges_index[0]):
                self.C[edges_index[0]] = True
            if not doesContainEdgeIndex(self.C, edges_index[1]):
                self.C[edges_index[1]] = True


"""
StateRecord - struktura dla przechowywania informacji w Miotle.
    edge_index - indeks krawędzi, jeżeli rekord służy jako liść
    inside_node_index - indeks krawędzi na lewo i maks. na prawo 
            od obecnego rekordu; jeżeli rekord nie jest liściem
"""


class StateRecord:
    def __init__(self, edge_index):
        self.edge_index = edge_index
        self.left = None
        self.right = None
        self.parent = None
        self.inside_node_index = edge_index
        self.height = 1


"""
Klasy dla przechowywania informacji w Podwójnie Lączonej Liście Krawędzi:
(odpowiednio dla wierzchołków, krawędzi oraz ścian)
"""


class VertexRecord:
    def __init__(self, x, y, incidental_edge=None):
        self.x = x
        self.y = y
        self.incidental_edge = incidental_edge

    def __iter__(self):
        yield self.x
        yield self.y

    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, other):
        return self.x == other.x and \
            self.y == other.y

    def __repr__(self):
        return str(tuple(self))


class HalfEdgeRecord:
    def __init__(self, beginning=None, twin_edge=None, incidental_faces=None,
                 next_edge=None, previous_edge=None):
        self.beginning = beginning
        self.twin_edge = twin_edge
        self.incidental_faces = []

        if incidental_faces != None:
            self.incidental_faces = [incidental_faces]

        self.next_edge = next_edge
        self.previous_edge = previous_edge
        self.G_vertex_id = None

    def destination(self):
        return self.twin_edge.beginning

    def __lt__(self, other):
        self_b = tuple(self.beginning)
        self_d = tuple(self.destination())
        other_d = tuple(other.destination())
        return orient(self_b, self_d, other_d) > 0

    def __iter__(self):
        yield tuple(self.beginning)
        yield tuple(self.destination())

    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, other):
        return self.beginning == other.beginning \
            and self.destination() == other.destination()

    def __repr__(self):
        return str(list(self))


class FaceRecord:
    def __init__(self, face_id, outer_edge=None, inner_edges=None, edge_to_outer_cycle=None):
        self.face_id = face_id
        self.outer_edge = outer_edge
        self.inner_edges = []
        if inner_edges != None:
            self.inner_edges = inner_edges.copy()

        self.edge_to_outer_cycle = edge_to_outer_cycle
        self.DCEL_id = None
        self.data = []

"""
vertices - słownik przyjmujący jako klucze krotki (x, y), będące współrzędnym wierzchołka, 
        natomiast wartościami są referencje do obiektów typu VertexRecord
edges - słownik, w którym kluczami są krotki ((x1, y1), (x2, y2)), a wartościami referencje 
        do jednej z półkrawędzi reprezentującej krawędź. Półkrawędzie są obiektami typu 
        HalfEdgeRecord. Aby sprawdzić czy krawędź znajduje się w słowniku lub wyciągnąć 
        jej wartość, musimy sprawdzić 2 ustawienia punktów ((x1, y1), (x2, y2)) oraz
        ((x2, y2), (x1, y1)), ponieważ nie wiemy która półkrawędź została wstawiona jako reprezentant.
faces - tablica zawierająca obiekty typu FaceRecord.
"""


class DoubleLinkedEdgeList:
    def __init__(self, vertices=None, edges=None, faces=None, precision=1e-05):
        if vertices:
            self.vertices = vertices
        else:
            self.vertices = {}

        if edges:
            self.edges = edges
        else:
            self.edges = {}

        if faces:
            self.faces = faces
        else:
            self.faces = []
        self.precision = precision
        self.next_free_face_index = len(self.faces)

    def joinList(self, joining_double_linked_list):

        if not isinstance(joining_double_linked_list, DoubleLinkedEdgeList):
            return Exception

        points_of_intersection = []

        for vertex in joining_double_linked_list.vertices.values():
            if self.doesContainVertex(vertex):
                self.mergeVertices(vertex, self.vertices[(vertex.x, vertex.y)])
                points_of_intersection.append((vertex.x, vertex.y))
            else:
                self.vertices[(vertex.x, vertex.y)] = vertex

        for edge in joining_double_linked_list.edges.values():
            if not self.doesContainEdge(edge):
                self.edges[((edge.beginning.x, edge.beginning.y),
                            (edge.twin_edge.beginning.x, edge.twin_edge.beginning.y))] = edge

        return points_of_intersection

    # Należy pamiętać, że ponieważ każda półkrawędź jest tak zorientowana, że ściana, którą ogranicza leży po jej
    # lewej stronie, to lewa półkrawędź będzie wychodząca z wierzchołka, natomiast prawa wchodząca do niego
    # edge_list zawierają referencje do półkrawędzi, natomiast slope_list wartości nachylenia krawędzi
    def mergeVertices(self, vertex1, vertex2):
        edge_list_1 = self.findIncidentalEdges(vertex1)
        edge_list_2 = self.findIncidentalEdges(vertex2)

        edge_list_1.reverse()
        edge_list_2.reverse()

        # edge_list_1 zawsze większa, bądź równa edge_list_2
        if len(edge_list_1) < len(edge_list_2):
            edge_list_1, edge_list_2 = edge_list_2, edge_list_1
            vertex1, vertex2 = vertex2, vertex1

        slope_list_1 = list(map(lambda x: self.getSlope(x, vertex1), edge_list_1))
        slope_list_2 = list(map(lambda x: self.getSlope(x, vertex2), edge_list_2))

        end, start = self.getEndPointsCircledTable(slope_list_1)

        for i in range(len(slope_list_2)):

            insert_index = self.binary_search_v2(slope_list_1, slope_list_2[i], start, end)

            # Krawędzie nachodzące na siebie
            if self.isSameVal(slope_list_1[insert_index-1][1], slope_list_2[i][1]) and self.isSameVal(slope_list_1[insert_index-1][0], slope_list_2[i][0]):
                length1 = self.getEdgeLength(edge_list_2[i])
                length2 = self.getEdgeLength(edge_list_1[insert_index-1])

                if self.isSameVal(length1, length2):
                    e = edge_list_1[insert_index-1]
                    self.delTotalEdge(edge_list_2[i])
                    self.edges[((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))] = e
                    self.edges[((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))].incidental_faces += edge_list_2[i].incidental_faces
                    continue

            elif insert_index < len(slope_list_1) and self.isSameVal(slope_list_1[insert_index][1], slope_list_2[i][1]) and self.isSameVal(slope_list_1[insert_index][0], slope_list_2[i][0]):
                length1 = self.getEdgeLength(edge_list_2[i])
                length2 = self.getEdgeLength(edge_list_1[insert_index])

                if self.isSameVal(length1, length2):
                    e = edge_list_1[insert_index]
                    self.delTotalEdge(edge_list_2[i])
                    self.edges[((e.beginning.x, e.beginning.y), (e.twin_edge.beginning.x, e.twin_edge.beginning.y))] = e
                    self.edges[((e.beginning.x, e.beginning.y),
                                (e.twin_edge.beginning.x, e.twin_edge.beginning.y))].incidental_faces += edge_list_2[
                        i].incidental_faces
                    continue

            slope_list_1 = slope_list_1[:insert_index] + [slope_list_2[i]] + slope_list_1[insert_index:]
            edge_list_1 = edge_list_1[:insert_index] + [edge_list_2[i]] + edge_list_1[insert_index:]

            if start < end:
                if insert_index == start:
                    start -= 1
                elif insert_index == end or insert_index == end + 1:
                    end += 1

            else:
                if insert_index <= start and insert_index <= end:
                    start += 1
                    end += 1
                elif insert_index == start:
                    if self.binary_search_compare_function(slope_list_2[i], slope_list_1[end]):
                        end += 1
                        start += 1
        edge_list_1.reverse()

        # Poprawne połączenie wszystkich półkrawędzi
        # tablica edge_list_1 została zapętlona za pomocą brania modułu z indeksów,
        # więc nie trzeba osobno rozpatrywać przypadków na krańcach tablicy
        for i in range(len(edge_list_1)):
            curr_edge_left = self.getOuterHalfEdge(edge_list_1[i], vertex1)
            curr_edge_right = self.getInnerHalfEdge(edge_list_1[i], vertex1)
            curr_edge_left.previous_edge = self.getInnerHalfEdge(edge_list_1[(i-1) % len(edge_list_1)], vertex1)
            curr_edge_left.previous_edge.next_edge = curr_edge_left
            curr_edge_right.next_edge = self.getOuterHalfEdge(edge_list_1[(i+1) % len(edge_list_1)], vertex1)
            curr_edge_right.next_edge.previous_edge = curr_edge_right

        return slope_list_1

    # Z założenia krawędź incydenta wychodzi z wierzchołka v
    # Zwraca tablicę z obiektami typu EdgeRecord
    def findIncidentalEdges(self, vertex):
        curr_edge = vertex.incidental_edge
        res = [curr_edge]
        curr_edge = curr_edge.twin_edge

        while curr_edge.next_edge != vertex.incidental_edge:
            curr_edge = curr_edge.next_edge
            res.append(curr_edge)
            curr_edge = curr_edge.twin_edge

        return res

    # zwraca krotkę 2 półkrawędzi, pomiędzy którymi należy włożyć krawędź (lewa, prawa)
    def findPlaceForInsert(self, vertex, edge_to_insert):
        insertion_slope = self.getSlope(edge_to_insert, vertex)
        curr_edge = vertex.incidental_edge

        while True:
            curr_slope = self.getSlope(curr_edge, vertex)
            # W lewo
            if self.binary_search_compare_function(insertion_slope, curr_slope):
                curr_edge_2 = curr_edge.previous_edge.twin_edge
                curr_slope_2 = self.getSlope(curr_edge_2, vertex)

                if self.binary_search_compare_function(insertion_slope, curr_slope_2) and \
                        (curr_slope_2[0] < insertion_slope[0] or
                         self.binary_search_compare_function(curr_slope_2, curr_slope)):
                    curr_edge = curr_edge_2
                else:
                    return (curr_edge_2, curr_edge)
            # W prawo
            else:
                curr_edge_2 = curr_edge.twin_edge.next_edge
                curr_slope_2 = self.getSlope(curr_edge_2, vertex)

                if self.binary_search_compare_function(insertion_slope, curr_slope_2) \
                        or self.binary_search_compare_function(curr_slope_2, curr_slope):
                    return (curr_edge, curr_edge_2)
                else:
                    curr_edge = curr_edge_2

    def binary_search_v2(self, arr, val, start, end):
        if start == end == -1:
            return 0

        if start == end:
            if self.binary_search_compare_function(val, arr[start]) == 0:
                return end
            else:
                return end + 1

        while end - start > 1 or start > end:
            if end == 0 and start == len(arr) - 1:
                break

            if start < end:
                mid = (start + end) // 2
            else:
                mid = ((start + len(arr) + end) // 2) % len(arr)

            if self.binary_search_compare_function(val, arr[mid]):
                start = mid
            else:
                end = mid

        if self.binary_search_compare_function(val, arr[start]) == 1 \
                and self.binary_search_compare_function(val, arr[end]) == 0:
            return end
        elif self.binary_search_compare_function(val, arr[start]) == 0:
            return start
        else:
            return end + 1

    # czy val1 jest później w kolejności niż val2
    # 0 pozycja w kolejności nierosnącej
    # 1 pozycja w kolejności niemalejącej
    def binary_search_compare_function(self, val1, val2):
        if val1[0] == val2[0]:
            if val1[1] >= val2[1]:
                return 1
            else:
                return 0
        elif val1[0] > val2[0]:
            return 0
        else:
            return 1

    def overlapAnyEdge(self, vertex, edges_array, edge, precision):
        edge_slope = self.getSlope(edge, vertex)[1]
        for arr_edge in edges_array:
            edge_slope_2 = self.getSlope(arr_edge, vertex)[1]
            if self.isSameVal(edge_slope, edge_slope_2):
                return arr_edge
        return None

    def delEdge(self, edge):
        point1 = edge.beginning
        point2 = edge.twin_edge.beginning

        try:
            self.edges[((point1.x, point1.y), (point2.x, point2.y))]
            del self.edges[((point1.x, point1.y), (point2.x, point2.y))]
            # self.edges.pop(((point1.x, point1.y), (point2.x, point2.y)))
        except KeyError:
            try:
                self.edges[((point2.x, point2.y), (point1.x, point1.y))]
                del self.edges[((point2.x, point2.y), (point1.x, point1.y))]
                # self.edges.pop(((point2.x, point2.y), (point1.x, point1.y)))
            except KeyError:
                pass

    # Bezpieczniejsza funkcja względem poprzedniej - usuwa oba warianty
    def delTotalEdge(self, edge):
        point1 = edge.beginning
        point2 = edge.twin_edge.beginning
        flag = False

        try:
            self.edges[((point1.x, point1.y), (point2.x, point2.y))]
            del self.edges[((point1.x, point1.y), (point2.x, point2.y))]
            flag = True
        except KeyError:
            pass

        try:
            self.edges[((point2.x, point2.y), (point1.x, point1.y))]
            del self.edges[((point2.x, point2.y), (point1.x, point1.y))]
            flag = True
        except KeyError:
            pass

        if not flag:
            print("!!!Failed to delete edge", ((point1.x, point1.y), (point2.x, point2.y)))

    def doesContainVertex(self, vertex):
        try:
            self.vertices[(vertex.x, vertex.y)]
            return True
        except KeyError:
            return False

    # Żeby sprawdzić czy krawędź jest w słowniku 'edges', musimy sprawdzić czy istnieje w nim
    # którakolwiek z jej półkrawędzi
    def doesContainEdge(self, edge):
        point1 = edge.beginning
        point2 = edge.twin_edge.beginning

        try:
            self.edges[((point1.x, point1.y), (point2.x, point2.y))]
            return True

        except KeyError:
            try:
                self.edges[((point2.x, point2.y), (point1.x, point1.y))]
                return True

            except KeyError:
                return False

    # Zwraca nachylenie przekazanego odcinka oraz ćwiartkę w której
    # się znajduje (jako środek układu uznajemy punkt)
    def getSlope(self, edge, vertex):

        # point1 to zawsze badany wierzchołek, z któego wychodzą krawędzie
        if self.isSamePoint(edge.beginning, vertex):
            point1 = vertex
            point2 = edge.twin_edge.beginning
        else:
            point1 = edge.twin_edge.beginning
            point2 = vertex

        delta_x = point2.x - point1.x
        delta_y = point2.y - point1.y

        if self.isSameVal(delta_x, 0):
            slope = sys.maxsize
        else:
            slope = delta_y / delta_x

        if self.isSameVal(delta_y, 0):
            if delta_x > 0:
                quarter = 1
            else:
                quarter = 3

        elif delta_y > 0:
            # sys.maxsize również działa
            if slope > 0:
                quarter = 1
            else:
                quarter = 4

        else:
            if slope > 0:
                quarter = 3
            else:
                quarter = 2

        return (quarter, slope)

    def getEdgeLength(self, edge):
        return np.sqrt(np.power(edge.twin_edge.beginning.x - edge.beginning.x, 2) + np.power(edge.twin_edge.beginning.y - edge.beginning.y, 2))

    def getEdge(self, edge):
        point1 = edge.beginning
        point2 = edge.twin_edge.beginning

        try:
            self.edges[((point1.x, point1.y), (point2.x, point2.y))]
            return self.edges[((point1.x, point1.y), (point2.x, point2.y))]

        except KeyError:
            try:
                self.edges[((point2.x, point2.y), (point1.x, point1.y))]
                return self.edges[((point2.x, point2.y), (point1.x, point1.y))]

            except KeyError:
                return None

    def getEdgeFromTuple(self, edge):
        try:
            self.edges[((edge[0][0], edge[0][1]), (edge[1][0], edge[1][1]))]
            return self.edges[((edge[0][0], edge[0][1]), (edge[1][0], edge[1][1]))]
        except KeyError:
            try:
                self.edges[((edge[1][0], edge[1][1]), (edge[0][0], edge[0][1]))]
                return self.edges[((edge[1][0], edge[1][1]), (edge[0][0], edge[0][1]))].twin_edge
            except KeyError:
                return None

    # Funkcje getInner/OuterHalfEdge przyjmują wierzchołek (VertexRecord)
    # oraz półkrawędź (HalfEdgeRecord)
    def getInnerHalfEdge(self, half_edge, vertex):
        if self.isSamePoint(half_edge.beginning, vertex):
            return half_edge.twin_edge
        else:
            return half_edge

    def getOuterHalfEdge(self, half_edge, vertex):
        if self.isSamePoint(half_edge.beginning, vertex):
            return half_edge
        else:
            return half_edge.twin_edge

    # zwraca (end, start)
    def getEndPointsCircledTable(self, arr):
        index = 0
        while index < len(arr) and arr[index][0] >= arr[(index+1) % len(arr)][0]:
            index += 1

        # Przypadek, w którym wszystkie odcinki są w tej samej ćwiartce
        # Wtedy index przejdzie po za tablicę i możemy zwrócić poczaek = 0
        # i koniec = len(arr) - 1
        if index == len(arr):
            index -= 1

        return (index, (index+1) % len(arr))

    def isSameVal(self, val1, val2):
        return abs(val2 - val1) < self.precision

    # Dla określenia czy 2 wierzchołki są tym samym wierzchołkiem
    # musimy porównać ich współrzędne  z pewną precyzją,
    # a nie ich referencje do obiektów
    def isSamePoint(self, vertex1, vertex2):
        return abs(vertex2.x - vertex1.x) < self.precision \
            and abs(vertex2.y - vertex1.y) < self.precision


# Jeżeli nie zawiera punktu, to rzuci wyjątek KeyError i zwróci False; jeżeli przejdzie, zwróci True
def doesContainEdgeIndex(dictionary, edge_index):
    try:
        dictionary[edge_index]
        return True
    except KeyError:
        return False


eps = 1e-5


def equals(x, y):
    return abs(x - y) < eps


def det3x33(a, b, c):
    return a[0] * b[1] + a[1] * c[0] + b[0] * c[1] - \
        b[1] * c[0] - a[0] * c[1] - a[1] * b[0]


def orient(a, b, c):
    det = det3x33(a, b, c)
    if det < -1*eps:
        return -1
    elif equals(det, 0):
        return 0
    else:
        return 1

# Znalezienie przecięcia krawędzi e oraz wierzchołka v:
#   1. e dzielimy na e' i e", względem wierzchołka v
#   2. krawędzie e' i e" rozdzielamy na półkrawędzie:
#       - 2 mają jako początek v i jako końce wierzchołki z e
#       - 2 mają jako początek wierzchołki z e i jako końce v
#       - łączymy odpowiednie półkrawędzie za pomocą pola 'twin'
#       (dodajemy 2 nowe rekordy do słownika z krawędziami)
#   3. naprawiamy pola 'next_edge' oraz 'previous_edge' - dla nowych krawędzi możemy użyć wskaźników z e,
#       jednak musimy pamiętać, żeby również naprawić krawędzie które są w następnie/poprzednie.
#       Jeżeli któryś z wierzchołków posiada jako krawędź incydentną krawędź, którą usuwamy, to również musimy ją
#       zamienić.
#   4. naprawiamy pola 'next_edge' oraz 'previous_edge' dla otoczenia v - musimy ustawić odpowiednie wskaźniki
#       dla nowych półkrawędzi oraz dla półkrawędzi obok nich. Możemy to zrobić znajdując porządek biegunowy krawędzi
#       wokół v. Każda półkrawędź wchodząca do v łączy się z pierwszą półkrawędzią wychodzącą z v zgodnie, z ruchem
#       wskazówek zegara. Dla półkrawędzi wychodzącej z v, jest odwrotnie.
#   5. usuwamy ze słownika e
#   Extras:
#       - są 3 możliwości dla przecięć odcinków z oddzielnych DoubleLinkedEdgeList:
#           a) punkt jest przecięciem odcinków w środku - wtedy musimy ustalić na nowo porządek biegunowy
#               i połączyć odpowiednie półkrawędzie (len(U) == len(L) == 0).
#               Mogą być to tylko 2 odcinki z różnych zbiorów, inaczej istniałby w tamtym miejscu wierzchołek
#               w którymś ze zbiorów.
#           b) punkt jest wierzchołkiem w jednych z DoubleLinkedEdgeList, natomiast krawędź z drugiej go przecina
#               (jeżeli przecinałoby go więcej krawędzi z drugiej listy, to w drugiej liście również byłby wierzchołkiem
#               i ten przypadek podchodziłby pod podpunkt c))
#           c) wierzchołek istnieje w obu DoubleLinkedEdgeListach i jego listy krawędzi został zmergowane przy
#               inicjalizacji wynikowej listy D. Najlepszym wyjściem byłoby ustalenie na nowo kolejności biegunowej
#               wokół wierzchołka i naprawienie ich wskaźników. Możemy wyeliminować ten przypadek, jeżeli wykonamy
#               odpowiednie operacje podczas mergowania list. Ten przypadek rozwiązujemy już na etapie tworzenia
#               nowej DoubleLinkedEdgeList.
#
#
# Naprawianie rekordów ścian:
#   1. Dla każdego cyklu wewnętrzego i zewnętrzego tworzymy wierzchołek w grafie G.
#   2. Na podstawie skrajnie lewych wierzchołków możemy stwierdzić czy cykl reprezentuje ścianę, czy dziurę w ścianie -
#       - ponieważ ściana incydentna zawsze leży po lewej stronie półkrawędzi, to możemy obliczyć kąt, jaki tworzy ten
#       wierzchołek z poprzednim i następnym - jeżeli kąt jest większy od 180 stopni, to jest to brzeg zewnętrzny,
#       a jeśli mniejszy, to brzeg wewnętrzny
#   3. Do grafu G dodajemy również jeden wierzchołek dla urojonego brzegu zewnętrznego brzegu nieograniczonej ściany.
#   4. Między dwoma wierzchołkami w G istnieje krawędź wtedy, gdy jeden jest brzegiem dziury, a drugi ma półkrawędź
#       bezpośrednio na lewo od skrajnie lewego weirzchołka tej dziury. Jeśli nie ma półkrawędzi na lewo od skrajnie
#       lewego wierzzchołka, to cykl jest dowiązany do wierzchołka urojonego brzegu nieograniczonej ściany.
#   5. Aby efektywnie znajdywać krawędzie między wierzchołkami, w rekordzie półkrawędzi dodajemy pole ze wskaźnikiem
#       do wierzchołka w grafie G, reprezentującego cykl, do którego ona należy.
#   6. Graf G będziemy na bierząco uzupełniać podczas zamiatania, ponieważ wtedy również znajdujemy krawędzie leżące
#       na lewo od siebie.
#
#
#   Koncepcja Algorytmu:
#   I. Podanie danych w wygodnej dla algorytmu zamiatania formie - surowej wersji DoubleLinkedEdgeList
#   II. Przetworzenie wierzchołków i krawędzi przez algorymt zamiatania, wyznaczenie przecięć oraz naprawienie
#       struktury DoubleLinkedEdgeList tak, że stanie się na powrót spójna.
#   III. Naprawienie rekordów ścian.
#
#
#
