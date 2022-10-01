# Algorytm zamiatania (Bentley-Ottmann Algorithm) dla nakładania się obszarów

import sys
import numpy as np
from visualizer import *
from classes import *


"""
Porządek '<' w drzewie AVL: p < q zachodzi wtw, gdy py > qy lub (py == qy i px < qx)
W tej relacji zdarzenie p zostanie przetworzone przed zdarzeniem q! Dlatego wcześniejsze zdarzenia będą umieszczane
na lewo od późniejszych zdarzeń.
"""


class AVLTree:
    def __init__(self, precision):
        self.tree_head = None
        self.E = None
        self.precision = precision

    # edges_values_list jest listą stworzoną z wartości słownika D.edges i zawiera obiekty typu VertexRecord
    # W DoubleLinkedEdgeList dla krawędzi używamy słownika, żeby utrzymywać tylko unilane wartości. W AVLTree natomiast
    # możemy przekazać sobie tablicę wartości, które są referencjami do obiektów typu HalfEdgeRecord i indeksować te
    # półkrawędzie po przypisanym im indeksie w tablicy.
    #
    # Przy dodawaniu nowych rekordów, sprawdzane jest najpierw czy rekord reprezentujący punkt (vertex.x, vertex.y)
    # przypadkiem już nie istnieje
    # Procedura wygląda następująco:
    #   1. Na podstawie 1 krawędzi stwórz z pierwszego punktu korzeń drzewa i dodaj 2 punkt
    #   2. Iterując od i=1 do i=n-1, dla obu końców odcinka sprawdź czy punkt jest już w drzewie:
    #       a) jeśli nie, to wstaw nowy rekord do drzewa
    #       b) jeśli tak, to dodaj do odpowiedniej tablicy w rekordzie reprezentującym ten punkt, indeks krawędzi
    def createAVLTree(self):
        if self.E == None:
            print("No Edge's list selected!")
            return

        n = len(self.E)
        node_type = isBeginning(self.E, self.E[0].beginning, 0, self.precision)

        head = QueueRecord(self.E[0].beginning.x, self.E[0].beginning.y, node_type, 0)

        # dodanie do drzewa drugiego końca odcinku
        # jeżeli curr_node == None, to mamy przypadek zdegenerowany, w którym odcinek jest punktem
        curr_node = self.searchFromPoint(head, self.E[0].twin_edge.beginning.x, self.E[0].twin_edge.beginning.y, self.precision)
        if curr_node == None:
            head = self.insertAVL(head, None, 0, self.E[0].twin_edge.beginning.x,
                                  self.E[0].twin_edge.beginning.y, (node_type+1) % 2, self.precision)
        else:
            if node_type == 0:
                curr_node.U.append(0)
            elif node_type == 1:
                curr_node.L[0] = True

        for i in range(1, n):
            curr_node = self.searchFromPoint(head, self.E[i].beginning.x,
                                             self.E[i].beginning.y, self.precision)

            # sprawdzenie czy 1 punkt w krawędzi jest początkiem czy końcem
            # drugi punkt będzie przeciwną opcją
            node_type = isBeginning(self.E, self.E[i].beginning, i, self.precision)
            if curr_node == None:
                head = self.insertAVL(head, None, i, self.E[i].beginning.x,
                                      self.E[i].beginning.y, node_type, self.precision)
            else:
                if node_type == 0:
                    curr_node.U.append(i)
                elif node_type == 1:
                    curr_node.L[i] = True

            curr_node = self.searchFromPoint(head, self.E[i].twin_edge.beginning.x,
                                             self.E[i].twin_edge.beginning.y, self.precision)
            if curr_node == None:
                head = self.insertAVL(head, None, i, self.E[i].twin_edge.beginning.x,
                                      self.E[i].twin_edge.beginning.y, (node_type+1) % 2, self.precision)
            else:
                if (node_type+1) % 2 == 0:
                    curr_node.U.append(i)
                elif (node_type+1) % 2 == 1:
                    curr_node.L[i] = True

        self.tree_head = head
        return head

    # Pytanie - czy (x, y) ma być wcześniej przetworzone niż node2
    # Opcja dająca wynik -1 została wprowadzona, aby można jej było również użyć przy szukaniu wierzchołka do usunięcia
    def compare(self, x, y, node2, precision):
        if isCloseVal(x, node2.x, precision) and isCloseVal(y, node2.y, precision):
            return -1

        round_place = int(-np.log10(precision))
        if round(y, round_place) > round(node2.y, round_place):
            return 1
        elif isCloseVal(y, node2.y, precision) and round(x, round_place) < round(node2.x, round_place):
            return 1
        else:
            return 0

    def getHeight(self, root):
        if root == None:
            return 0
        else:
            return root.height

    def getBalance(self, root):
        if root == None:
            return 0
        return self.getHeight(root.left) - self.getHeight(root.right)

    def getMinValueNode(self, root):
        if root == None or root.left == None:
            return root
        return self.getMinValueNode(root.left)

    def getMaxValueNode(self, root):
        if root == None or root.right == None:
            return root
        return self.getMaxValueNode(root.right)

    def hasQueueEnded(self):
        if self.tree_head == None:
            return True
        else:
            return False

    # Funkcja zwracająca punkt przecięcia 2 odcinków. Jeżeli odcinki się nie przecinają, to zwraca None
    # segment1, segment2 są obiektami typu HalfEdgeRecord
    def findIntersection(self, segment1, segment2, precision):

        delta1_y = segment1.twin_edge.beginning.y - segment1.beginning.y
        delta1_x = segment1.twin_edge.beginning.x - segment1.beginning.x
        delta2_y = segment2.twin_edge.beginning.y - segment2.beginning.y
        delta2_x = segment2.twin_edge.beginning.x - segment2.beginning.x

        # Dwie pionowe linie
        if abs(delta1_x) < precision and abs(delta2_x) < precision:
            return None

        # Dwie poziome linie
        if abs(delta1_y) < precision and abs(delta2_y) < precision:
            return None

        # jeżeli odcinek 2 jest pionowy, to dla zaoszczędzenia pisania kodu zamieniamy oznaczenia odcinka 1 z 2,
        # dzięki czemu, jeżeli któryś z odcinków jest pionowy, to tylko odcinek 1
        if abs(delta2_x) < precision:
            delta1_x, delta2_x = delta2_x, delta1_x
            delta1_y, delta2_y = delta2_y, delta1_y
            segment1, segment2 = segment2, segment1

        # odciniek 1 jest pionowy
        if abs(delta1_x) < precision:

            if segment1.beginning.y < segment1.twin_edge.beginning.y:
                bottom_point = segment1.beginning
                upper_point = segment1.twin_edge.beginning
            else:
                bottom_point = segment1.twin_edge.beginning
                upper_point = segment1.beginning

            side_0 = det2x2(bottom_point, upper_point,
                            segment2.beginning, precision)
            side_1 = det2x2(bottom_point, upper_point,
                            segment2.twin_edge.beginning, precision)

            # jeżeli jeden z punktów odcinka 2 leży na odcinku 1, to istnieje przecięcie
            if side_0 == 0 or side_1 == 0:

                # ponieważ odcinek 1 jest pionowy, to x jest ustalony
                x = segment1.beginning.x
                slope2 = (segment2.twin_edge.beginning.y - segment2.beginning.y) \
                    / (segment2.twin_edge.beginning.x - segment2.beginning.x)
                b2 = segment2.beginning.y - slope2 * segment2.beginning.x

                y = slope2 * x + b2

                if isPointBetween(segment1.beginning, segment1.twin_edge.beginning, x, y, precision):
                    return (x, y)
                else:
                    return None

            # jeżeli punkty odcinka 2 leżą po przeciwnych stronach odcinka 1, to istnieje przecięcie
            elif side_0 != side_1:

                # ponieważ odcinek 1 jest pionowy, to x jest ustalony
                x = segment1.beginning.x
                slope2 = (segment2.twin_edge.beginning.y - segment2.beginning.y) \
                    / (segment2.twin_edge.beginning.x - segment2.beginning.x)
                b2 = segment2.beginning.y - slope2 * segment2.beginning.x

                y = slope2 * x + b2

                if isPointBetween(segment1.beginning, segment1.twin_edge.beginning, x, y, precision):
                    return (x, y)
                else:
                    return None

            else:
                return None

        else:

            if segment1.beginning.y < segment1.twin_edge.beginning.y:
                bottom_point = segment1.beginning
                upper_point = segment1.twin_edge.beginning
            else:
                bottom_point = segment1.twin_edge.beginning
                upper_point = segment1.beginning

            side_0 = det2x2(bottom_point, upper_point,
                            segment2.beginning, precision)
            side_1 = det2x2(bottom_point, upper_point,
                            segment2.twin_edge.beginning, precision)

            # jeżeli odcinki są współliniowe lub przecinają się
            if side_0 == 0 or side_1 == 0 or side_0 != side_1:

                # (y2 - y1) / (x2 - x1)
                slope1 = delta1_y / delta1_x
                slope2 = delta2_y / delta2_x

                if isCloseVal(slope1, slope2, precision):
                    return None

                # b = y - ax
                b1 = segment1.beginning.y - slope1 * segment1.beginning.x
                b2 = segment2.beginning.y - slope2 * segment2.beginning.x

                # x0 = (b2 - b1) / (a1 - a2)
                x = (b2 - b1) / (slope1 - slope2)

                # y0 = a * x0 + b
                y = slope1 * x + b1

                if isPointBetween(segment1.beginning, segment1.twin_edge.beginning, x, y, precision):
                    return (x, y)
                else:
                    return None
            else:
                return None

    # Operacja szukająca wśród wierzchołków punktu (x, y). Jeśli go znajdzie, zwraca do niego referencję.
    # W przeciwnym razie zwróci None.
    def searchFromPoint(self, root, x, y, precision):
        if root == None:
            return None
        if isCloseVal(x, root.x, precision) and isCloseVal(y, root.y, precision):
            return root
        elif self.compare(x, y, root, precision) == 1:
            return self.searchFromPoint(root.left, x, y, precision)
        else:
            return self.searchFromPoint(root.right, x, y, precision)

    def findPredecessor(self, node):

        # jeżeli da się iść w lewo z node'a, to idź do dziecka w lewym poddrzewie, najbardziej po prawej stronie
        if node.left != None:
            node = node.left
            while node.right != None:
                node = node.right
        else:
            parent_node = node.parent
            # jeżeli node jest korzeniem drzewa i nie ma lewego dziecka, to nie istnieje poprzednik
            if parent_node == None:
                return None

            # jeżeli parent istnieje i node jest jego prawym dzieckiem, to parent jest jego poprzednikiem
            if node == parent_node.right:
                node = parent_node
            else:
                # cofaj się tak długo, aż znajdziesz pierwszego parenta,
                # dla którego obecny node jest jego prawym dzieckiem
                while parent_node != None and parent_node.right != node:
                    node = parent_node
                    parent_node = parent_node.parent
                node = parent_node

        return node

    def findConsequent(self, node):

        if node.right != None:
            node = node.right
            while node.left != None:
                node = node.left
        else:
            parent_node = node.parent
            if parent_node == None:
                return None

            if node == parent_node.left:
                node = parent_node
            else:
                while parent_node != None and parent_node.left != node:
                    node = parent_node
                    parent_node = parent_node.parent
                node = parent_node

        return node

    # Kopiuje informacje z node2 do node1
    def copyNode(self, node1, node2):
        node1.x = node2.x
        node1.y = node2.y

        node1.record_type = node2.record_type
        node1.U = node2.U
        node1.L = node2.L
        node1.C = node2.C

    # Operacja insertAVL jest wykonywana przy założeniu, że punkt (x, y) nie jest duplikatem innego punktu w drzewie
    def insertAVL(self, root, parent, edge_index, x, y, node_type, precision):
        if root == None:
            new_node = QueueRecord(x, y, node_type, edge_index)
            new_node.parent = parent
            return new_node

        elif self.compare(x, y, root, precision) == 1:
            root.left = self.insertAVL(
                root.left, root, edge_index, x, y, node_type, precision)
        else:
            root.right = self.insertAVL(
                root.right, root, edge_index, x, y, node_type, precision)

        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        balance = self.getBalance(root)

        # LL - rotation
        if balance > 1 and self.compare(x, y, root.left, precision) == 1:
            return self.rightRotate(root)

        # RR - rotation
        if balance < -1 and self.compare(x, y, root.right, precision) == 0:
            return self.leftRotate(root)

        # LR - rotation
        if balance > 1 and self.compare(x, y, root.left, precision) == 0:
            root.left = self.leftRotate(root.left)
            return self.rightRotate(root)

        # RL - rotation
        if balance < -1 and self.compare(x, y, root.right, precision) == 1:
            root.right = self.rightRotate(root.right)
            return self.leftRotate(root)

        return root

    def deleteAVL(self, root, parent, x, y, precision):
        if root == None:
            return root

        elif self.compare(x, y, root, precision) == 1:
            root.left = self.deleteAVL(root.left, root, x, y, precision)
        elif self.compare(x, y, root, precision) == 0:
            root.right = self.deleteAVL(root.right, root, x, y, precision)

        else:
            # temp == None, jeżeli element do usunięcia jest liściem
            if root.left == None:
                temp = root.right
                root.parent = None
                root = None

                if temp != None:
                    temp.parent = parent
                return temp

            elif root.right == None:
                temp = root.left
                root.parent = None
                root = None

                if temp != None:
                    temp.parent = parent
                return temp

            # znajdź następnika
            temp = self.getMinValueNode(root.right)

            # Przekopiuj jego wartości do roota
            self.copyNode(root, temp)

            # Usuń następnika z poddrzewa (usunięcie będzie na pewno jedną z opcji w ifach,
            # więc będzie prostym przepięciem)
            root.right = self.deleteAVL(
                root.right, root, temp.x, temp.y, precision)

        if root == None:
            return root

        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))
        balance = self.getBalance(root)

        # LL - rotation
        if balance > 1 and self.getBalance(root.left) >= 0:
            return self.rightRotate(root)

        # RR - rotation
        if balance < -1 and self.getBalance(root.right) <= 0:
            return self.leftRotate(root)

        # LR - rotation
        if balance > 1 and self.getBalance(root.left) < 0:
            root.left = self.leftRotate(root.left)
            return self.rightRotate(root)

        # RL - rotation
        if balance < -1 and self.getBalance(root.right) > 0:
            root.right = self.rightRotate(root.right)
            return self.leftRotate(root)

        return root

    def leftRotate(self, z):
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        if y != None:
            y.parent = z.parent
        if z != None:
            z.parent = y
        if T2 != None:
            T2.parent = z

        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        return y

    def rightRotate(self, z):
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        if y != None:
            y.parent = z.parent
        if z != None:
            z.parent = y
        if T3 != None:
            T3.parent = z

        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        return y


class AVLTreeBroomState:
    def __init__(self):
        self.tree_head = None

    def getHeight(self, root):
        if root == None:
            return 0
        else:
            return root.height

    def getBalance(self, root):
        if root == None:
            return 0
        return self.getHeight(root.left) - self.getHeight(root.right)

    def leftRotate(self, z):
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        if y != None:
            y.parent = z.parent
        if z != None:
            z.parent = y
        if T2 != None:
            T2.parent = z

        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        return y

    def rightRotate(self, z):
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        if y != None:
            y.parent = z.parent
        if z != None:
            z.parent = y
        if T3 != None:
            T3.parent = z

        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        return y

    def search(self, root, edge_index, ending_point, E, precision):

        if root.left == None and root.right == None and root.edge_index == edge_index:
            return root

        if root.inside_node_index == edge_index:
            root = root.left
            while root.right != None:
                root = root.right
            return root

        if isBeginning(E, E[root.inside_node_index].beginning, root.inside_node_index, precision) == 0:
            side = det2x2(E[root.inside_node_index].beginning, E[root.inside_node_index].twin_edge.beginning,
                          ending_point, precision)

        else:
            side = det2x2(E[root.inside_node_index].twin_edge.beginning, E[root.inside_node_index].beginning,
                          ending_point, precision)

        if side == 0:
            slope_internal, slope_insert = None, None
            if isCloseVal(E[root.inside_node_index].beginning.x, E[root.inside_node_index].twin_edge.beginning.x, precision):
                slope_internal = sys.maxsize
            elif isCloseVal(E[root.inside_node_index].beginning.y, E[root.inside_node_index].twin_edge.beginning.y, precision):
                slope_internal = 0
            else:
                if isBeginning(E, E[root.inside_node_index].beginning, root.inside_node_index, precision) == 0:
                    point_end_internal = E[root.inside_node_index].beginning
                    point_start_internal = E[root.inside_node_index].twin_edge.beginning
                else:
                    point_end_internal = E[root.inside_node_index].twin_edge.beginning
                    point_start_internal = E[root.inside_node_index].beginning

            if isCloseVal(E[edge_index].beginning.x, E[edge_index].twin_edge.beginning.x, precision):
                slope_insert = sys.maxsize
            elif isCloseVal(E[edge_index].beginning.y, E[edge_index].twin_edge.beginning.y, precision):
                slope_insert = 0
            else:
                if isBeginning(E, E[edge_index].beginning, edge_index, precision) == 0:
                    point_end_insert = E[edge_index].beginning
                    point_start_insert = E[edge_index].twin_edge.beginning
                else:
                    point_end_insert = E[edge_index].twin_edge.beginning
                    point_start_insert = E[edge_index].beginning

            if slope_internal == None:
                slope_internal = (point_end_internal.y - point_start_internal.y) / (
                    point_end_internal.x - point_start_internal.x)

            if slope_insert == None:
                slope_insert = (point_end_insert.y - point_start_insert.y) / (
                    point_end_insert.x - point_start_insert.x)

            flag = 0
            # 1 - po lewej
            # 2 - po prawej
            if slope_internal == sys.maxsize:
                if slope_insert > 0:
                    flag = 1
                elif slope_insert <= 0:
                    flag = 2

            elif slope_insert == sys.maxsize:
                if slope_internal > 0:
                    flag = 2
                elif slope_internal <= 0:
                    flag = 1

            elif slope_internal == 0:
                flag = 1

            elif slope_insert == 0:
                flag = 2

            elif (slope_internal > 0 and slope_insert > 0) or (slope_internal < 0 and slope_insert < 0):
                if slope_internal > slope_insert:
                    flag = 1
                else:
                    flag = 2

            elif slope_internal > 0 and slope_insert < 0:
                flag = 2
            elif slope_internal < 0 and slope_insert > 0:
                flag = 1

        if side == 1 or (side == 0 and flag == 1):
            return self.search(root.left, edge_index, ending_point, E, precision)
        else:
            return self.search(root.right, edge_index, ending_point, E, precision)

    def searchWithDFSHelp(self, root, edge_index, ending_point, E, precision):

        if root.left == None and root.right == None and root.edge_index == edge_index:
            return root

        if root.inside_node_index == edge_index:
            root = root.left
            while root.right != None:
                root = root.right
            return root

        if isBeginning(E, E[root.inside_node_index].beginning, root.inside_node_index, precision) == 0:
            side = det2x2(E[root.inside_node_index].beginning, E[root.inside_node_index].twin_edge.beginning,
                          ending_point, precision)
        else:
            side = det2x2(E[root.inside_node_index].twin_edge.beginning, E[root.inside_node_index].beginning,
                          ending_point, precision)

        if side == 0:
            return self.DFSEmergency(root, edge_index, precision)
        if side == 1:
            return self.search(root.left, edge_index, ending_point, E, precision)
        else:
            return self.search(root.right, edge_index, ending_point, E, precision)

    def DFSEmergency(self, root, edge_index, precision):
        if root.left == None and root.right == None and root.edge_index == edge_index:
            return root

        if root.left != None:
            left = self.DFSEmergency(root.left, edge_index, precision)
            if left != None:
                return left

        if root.right != None:
            right = self.DFSEmergency(root.right, edge_index, precision)
            if right != None:
                return right

        return None

    def findPredecessor(self, root):
        root = root.left

        while root.right != None:
            root = root.right

        return root

    def findLeft(self, root):
        if root.parent == None:
            return None

        while root.parent.right != root:
            root = root.parent
            if root.parent == None:
                return None

        root = root.parent.left
        while root.right != None:
            root = root.right

        return root

    def findRight(self, root):
        if root.parent == None:
            return None

        while root.parent.left != root:
            root = root.parent
            if root.parent == None:
                return None

        root = root.parent.right
        while root.left != None:
            root = root.left

        return root

    def insertAVL(self, root, edge_index, starting_point, E, precision):

        if self.tree_head == None:
            return StateRecord(edge_index)

        if isBeginning(E, E[root.inside_node_index].beginning, root.inside_node_index, precision) == 0:
            side = det2x2(E[root.inside_node_index].beginning, E[root.inside_node_index].twin_edge.beginning,
                          starting_point, precision)
        else:
            side = det2x2(E[root.inside_node_index].twin_edge.beginning, E[root.inside_node_index].beginning,
                          starting_point, precision)

        if side == 0:
            slope_internal, slope_insert = None, None
            if isCloseVal(E[root.inside_node_index].beginning.x, E[root.inside_node_index].twin_edge.beginning.x, precision):
                slope_internal = sys.maxsize
            elif isCloseVal(E[root.inside_node_index].beginning.y, E[root.inside_node_index].twin_edge.beginning.y, precision):
                slope_internal = 0
            else:
                if isBeginning(E, E[root.inside_node_index].beginning, root.inside_node_index, precision) == 0:
                    point_end_internal = E[root.inside_node_index].beginning
                    point_start_internal = E[root.inside_node_index].twin_edge.beginning
                else:
                    point_end_internal = E[root.inside_node_index].twin_edge.beginning
                    point_start_internal = E[root.inside_node_index].beginning

            if isCloseVal(E[edge_index].beginning.x, E[edge_index].twin_edge.beginning.x, precision):
                slope_insert = sys.maxsize
            elif isCloseVal(E[edge_index].beginning.y, E[edge_index].twin_edge.beginning.y, precision):
                slope_insert = 0
            else:
                if isBeginning(E, E[edge_index].beginning, edge_index, precision) == 0:
                    point_end_insert = E[edge_index].beginning
                    point_start_insert = E[edge_index].twin_edge.beginning
                else:
                    point_end_insert = E[edge_index].twin_edge.beginning
                    point_start_insert = E[edge_index].beginning

            if slope_internal == None:
                slope_internal = (point_end_internal.y - point_start_internal.y) / \
                                 (point_end_internal.x - point_start_internal.x)

            if slope_insert == None:
                slope_insert = (point_end_insert.y - point_start_insert.y) / \
                               (point_end_insert.x - point_start_insert.x)

            flag = 0
            # 1 - po lewej
            # 2 - po prawej
            if slope_internal == sys.maxsize:
                if slope_insert > 0:
                    flag = 1
                elif slope_insert <= 0:
                    flag = 2

            elif slope_insert == sys.maxsize:
                if slope_internal > 0:
                    flag = 2
                elif slope_internal <= 0:
                    flag = 1

            elif slope_internal == 0:
                flag = 1

            elif slope_insert == 0:
                flag = 2

            elif (slope_internal > 0 and slope_insert > 0) or (slope_internal < 0 and slope_insert < 0):
                if slope_internal > slope_insert:
                    flag = 1
                else:
                    flag = 2

            elif slope_internal > 0 and slope_insert < 0:
                flag = 2
            elif slope_internal < 0 and slope_insert > 0:
                flag = 1

        if root.left == None and root.right == None:

            if side == 1 or (side == 0 and flag == 1):
                new_internal_node = StateRecord(root.edge_index)
                new_node = StateRecord(edge_index)

                new_internal_node.parent = root.parent
                new_internal_node.right = root
                root.parent = new_internal_node
                new_internal_node.left = new_node
                new_node.parent = new_internal_node
                new_internal_node.inside_node_index = edge_index
                new_internal_node.height = 2

                return new_internal_node

            elif side == 2 or (side == 0 and flag == 2):
                new_internal_node = StateRecord(edge_index)
                new_node = StateRecord(edge_index)

                new_internal_node.parent = root.parent
                new_internal_node.right = new_node
                new_node.parent = new_internal_node
                new_internal_node.left = root
                root.parent = new_internal_node
                new_internal_node.inside_node_index = root.edge_index
                new_internal_node.height = 2

                return new_internal_node

        elif side == 1 or (side == 0 and flag == 1):
            root.left = self.insertAVL(
                root.left, edge_index, starting_point, E, precision)
            root.inside_node_index = self.findPredecessor(
                root).inside_node_index
        else:
            root.right = self.insertAVL(
                root.right, edge_index, starting_point, E, precision)

        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        balance = self.getBalance(root)

        # LL - rotation
        if balance > 1 and self.getBalance(root.left) >= 0:
            return self.rightRotate(root)

        # RR - rotation
        if balance < -1 and self.getBalance(root.right) <= 0:
            return self.leftRotate(root)

        # LR - rotation
        if balance > 1 and self.getBalance(root.left) < 0:
            root.left = self.leftRotate(root.left)
            return self.rightRotate(root)

        # RL - rotation
        if balance < -1 and self.getBalance(root.right) > 0:
            root.right = self.rightRotate(root.right)
            return self.leftRotate(root)

        return root

    # Możemy usunąć dowolną krawędź przecinającą punkt p, ponieważ w pierwszym kroku przetwarzania zdarzenia
    # musimy usunąć wszystkie krawędzie w L i C, więc każda taka krawędź na pewno się tam znalazła
    def deleteAVLbyPoint(self, root, broom_point, E, precision):

        if root == None:
            return root, None

        if isBeginning(E, E[root.edge_index].beginning, root.edge_index, precision) == 0:
            side_mid = det2x2(E[root.edge_index].beginning, E[root.edge_index].twin_edge.beginning,
                              broom_point, precision)
        else:
            side_mid = det2x2(E[root.edge_index].twin_edge.beginning, E[root.edge_index].beginning,
                              broom_point, precision)


        # Usunięto (and root.parent == None)
        if side_mid == 0 and root.left == None and root.right == None:
            return None, root.edge_index

        if isBeginning(E, E[root.left.edge_index].beginning, root.left.edge_index, precision) == 0:
            side_left = det2x2(E[root.left.edge_index].beginning, E[root.left.edge_index].twin_edge.beginning,
                               broom_point, precision)
        else:
            side_left = det2x2(E[root.left.edge_index].twin_edge.beginning, E[root.left.edge_index].beginning,
                               broom_point, precision)

        if isBeginning(E, E[root.right.edge_index].beginning, root.right.edge_index, precision) == 0:
            side_right = det2x2(E[root.right.edge_index].beginning, E[root.right.edge_index].twin_edge.beginning,
                                broom_point, precision)
        else:
            side_right = det2x2(E[root.right.edge_index].twin_edge.beginning, E[root.right.edge_index].beginning,
                                broom_point, precision)


        if side_left == 0 and root.left.left == None and root.left.right == None:
            if root.parent == None:
                return root.right, root.left.edge_index

            root.right.parent = root.parent
            return root.right, root.left.edge_index

        elif side_right == 0 and root.right.left == None and root.right.right == None:
            if root.parent == None:
                return root.left, root.right.edge_index

            root.left.parent = root.parent
            return root.left, root.right.edge_index

        if isBeginning(E, E[root.inside_node_index].beginning, root.inside_node_index, precision) == 0:
            side = det2x2(E[root.inside_node_index].beginning, E[root.inside_node_index].twin_edge.beginning,
                          broom_point, precision)
        else:
            side = det2x2(E[root.inside_node_index].twin_edge.beginning, E[root.inside_node_index].beginning,
                          broom_point, precision)


        if side == 1 or side == 0:
            root.left, deleted_node_index = self.deleteAVLbyPoint(
                root.left, broom_point, E, precision)
            root.inside_node_index = self.findPredecessor(
                root).inside_node_index
        elif side == 2:
            root.right, deleted_node_index = self.deleteAVLbyPoint(
                root.right, broom_point, E, precision)

        if root == None:
            return root, deleted_node_index

        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))
        balance = self.getBalance(root)

        # LL - rotation
        if balance > 1 and self.getBalance(root.left) >= 0:
            return self.rightRotate(root), deleted_node_index

        # RR - rotation
        if balance < -1 and self.getBalance(root.right) <= 0:
            return self.leftRotate(root), deleted_node_index

        # LR - rotation
        if balance > 1 and self.getBalance(root.left) < 0:
            root.left = self.leftRotate(root.left)
            return self.rightRotate(root), deleted_node_index

        # RL - rotation
        if balance < -1 and self.getBalance(root.right):
            root.right = self.rightRotate(root.right)
            return self.leftRotate(root), deleted_node_index

        return root, deleted_node_index


# funkcja zwracająca 0, jeżeli przekazany punkt jest początkiem odcinka i 1, jeżeli nie jest
def isBeginning(E, point1, edge_index, precision):

    if E[edge_index].beginning != point1:
        point2 = E[edge_index].beginning
    else:
        point2 = E[edge_index].twin_edge.beginning

    round_place = int(-np.log10(precision))
    if round(point1.y, round_place) > round(point2.y, round_place):
        return 0
    elif abs(point1.y - point2.y) < precision and round(point1.x, round_place) < round(point2.x, round_place):
        return 0
    else:
        return 1


def findOrder(D, Q, U, L, C, E, mapped_edges, ending_point, points_of_intersection, precision):
    res_negative = []
    res_positive = []
    res_middle = []
    res_end = []

    for i in range(len(U)):

        if D.isSamePoint(E[U[i]].beginning, ending_point):
            point1 = E[U[i]].beginning
            point2 = E[U[i]].twin_edge.beginning
        else:
            point1 = E[U[i]].twin_edge.beginning
            point2 = E[U[i]].beginning

        if isCloseVal(point2.x, point1.x, precision):
            res_middle.append(U[i])
            continue

        slope = (point2.y - point1.y) / (point2.x - point1.x)

        if abs(slope) < precision:
            res_end.append(U[i])
        elif slope > 0:
            res_positive.append((slope, U[i]))
        else:
            res_negative.append((slope, U[i]))

    for C_index in C.keys():

        if doesContainEdgeIndex(L, C_index):
            continue

        # Jeżeli 2 krawędzie na siebie nachodzą
        edge_overlap = overlapAnyEdge(D, Q, ending_point, U, C_index, precision)
        if edge_overlap != None:
            length1 = getEdgeLength(E[C_index])
            length2 = getEdgeLength(E[edge_overlap])


            # Jeżeli są takiej samej długości, to tylko, jeżeli ich oryginalne krawędzie zaczynały się w różnych
            # punktach. Nie ma w tym momencie różnicy którą z nich dodamy, ponieważ są praktycznie dokładnie takie same.
            # Zostawimy więc tą, która już jest w tablicy, natomiast nowej nie dodamy. Musimy jedynie zmienić w jej
            # wierzchołku końcowym tablicę L, aby poinformować ją, że ta krawędź już się tam nie kończy.
            if D.isSameVal(length1, length2):
                if isBeginning(E, E[C_index].beginning, C_index, precision) == 0:
                    ending_vertex = E[C_index].twin_edge.beginning
                else:
                    ending_vertex = E[C_index].beginning

                # if D.isSamePoint(E[C_index].beginning, ending_point):
                #     delEdgeFromVertex(E[C_index])
                # else:
                #     delEdgeFromVertex(E[C_index].twin_edge)

                delEdgeFromBothVertices(E[C_index])

                vertex_to_change = Q.searchFromPoint(Q.tree_head, ending_vertex.x, ending_vertex.y, precision)
                del vertex_to_change.L[C_index]

                # Usunięcie krawędzi z DCEL i dodanie na nowo
                # Ten krok jest po to, że w tym przypadku mamy 2 krawędzie tej samej długości, zaczynające się i koń-
                # czące w tych samych wierzchołkach, przez co są one praktycznie nierozróżnialne (oprócz adresu w pa-
                # mięci). Jedna z krawędzi będzie krawędzią z jednego ze zbiorów wejściowych, a druga utworzoną podczas
                # handlowania przecięcia (w tym momencie w DCEL zostanie nadpisana krawędź o tych samych punktach).
                # Zostawiamy tę krawędź, która była w zbiorze wejściowym, więc potrzebujemy ją przywrócić do DCEL.
                D.delTotalEdge(E[C_index])
                D.edges[((E[edge_overlap].beginning.x, E[edge_overlap].beginning.y), (E[edge_overlap].twin_edge.beginning.x, E[edge_overlap].twin_edge.beginning.y))] = E[edge_overlap]

                continue

            # Zależnie od tego, która krawędź jest krótsza, wykonujemy zamienne operacje
            elif length1 < length2:
                delEdgeIndexWhileFindingOrder(res_negative, res_positive, res_middle, res_end, edge_overlap)
                handleOverlapingEdges(D, Q, mapped_edges, C_index, edge_overlap, ending_point, points_of_intersection, precision)
            else:
                handleOverlapingEdges(D, Q, mapped_edges, edge_overlap, C_index, ending_point, points_of_intersection, precision)
                continue

        if D.isSamePoint(E[C_index].beginning, ending_point):
            point1 = E[C_index].beginning
            point2 = E[C_index].twin_edge.beginning
        else:
            point1 = E[C_index].twin_edge.beginning
            point2 = E[C_index].beginning

        if isCloseVal(point2.x, point1.x, precision):
            res_middle.append(C_index)
        else:
            slope = (point2.y - point1.y) / (point2.x - point1.x)
            if abs(slope) < precision:
                res_end.append(C_index)
            elif slope > 0:
                res_positive.append((slope, C_index))
            else:
                res_negative.append((slope, C_index))

    if len(res_positive) != 0:
        res_positive.sort(key=lambda x: x[0])
    if len(res_negative) != 0:
        res_negative.sort(key=lambda x: x[0])

    return [x[1] for x in res_positive] + res_middle + [x[1] for x in res_negative] + res_end


def delEdgeIndexWhileFindingOrder(neg, pos, mid, end, del_index):
    for i in range(len(neg)):
        if neg[i][1] == del_index:
            neg.pop(i)
            return

    for i in range(len(pos)):
        if pos[i][1] == del_index:
            pos.pop(i)
            return

    for i in range(len(mid)):
        if mid[i] == del_index:
            mid.pop(i)
            return

    for i in range(len(end)):
        if end[i] == del_index:
            end.pop(i)
            return


# Założenie - podana półkrawędź ma początek w rozważanym wierzchołku
# Funkcja usuwająca z wierzchołka obie półkrawędzie za pomocą rozłączania wskaźników
def delEdgeFromVertex(half_edge):
    half_edge.twin_edge.next_edge.previous_edge = half_edge.previous_edge
    half_edge.previous_edge.next_edge = half_edge.twin_edge.next_edge

    half_edge.next_edge = None
    half_edge.previous_edge = None


def delEdgeFromBothVertices(half_edge):
    half_edge.twin_edge.next_edge.previous_edge = half_edge.previous_edge
    half_edge.previous_edge.next_edge = half_edge.twin_edge.next_edge

    half_edge.next_edge.previous_edge = half_edge.twin_edge.previous_edge
    half_edge.twin_edge.previous_edge.next_edge = half_edge.next_edge

    half_edge.next_edge = None
    half_edge.previous_edge = None
    half_edge.twin_edge.next_edge = None
    half_edge.twin_edge.previous_edge = None


# 'Zielona' - odnosi się do dolnej krawędzi, która powstaje po podziale dłuższej krawędzi w punkcie końcowym
# krótszej krawędzi. Nazwa wzięła się z rysunku, na którym rozpisywałem tę funkcję.
def handleOverlapingEdges(D, Q, mapped_edges, shorter_edge_index, longer_edge_index, event, points_of_intersection, precision):
    # shorter_edge - fioletowa, longer_edge - zielona

    if isBeginning(Q.E, Q.E[shorter_edge_index].beginning, shorter_edge_index, precision) == 0:
        ending_point_1 = Q.E[shorter_edge_index].twin_edge.beginning
    else:
        ending_point_1 = Q.E[shorter_edge_index].beginning

    points_of_intersection.append((ending_point_1.x, ending_point_1.y))

    if isBeginning(Q.E, Q.E[longer_edge_index].beginning, longer_edge_index, precision) == 0:
        ending_point_2 = Q.E[longer_edge_index].twin_edge.beginning
        long_segment = Q.E[longer_edge_index]
    else:
        ending_point_2 = Q.E[longer_edge_index].beginning
        long_segment = Q.E[longer_edge_index].twin_edge

    vertex = D.vertices[(event.x, event.y)]

    bottom_edge_1 = HalfEdgeRecord(beginning=ending_point_1)
    bottom_edge_2 = HalfEdgeRecord(beginning=ending_point_2)

    bottom_edge_1.twin_edge = bottom_edge_2
    bottom_edge_2.twin_edge = bottom_edge_1

    # Dodanie nowej krawędzi do struktur
    D.edges[((bottom_edge_1.beginning.x, bottom_edge_1.beginning.y), (bottom_edge_2.beginning.x, bottom_edge_2.beginning.y))] = bottom_edge_1
    Q.E.append(bottom_edge_1)

    # Dodanie nowej krawędzi do mapped_edges
    mapped_edges.append(len(mapped_edges))
    mapped_edges[longer_edge_index] = mapped_edges[-1]

    # Naprawienie wskaźników na końcu krawędzi (tylko w w3)
    bottom_edge_1.next_edge = long_segment.next_edge
    long_segment.next_edge.previous_edge = bottom_edge_1
    bottom_edge_2.previous_edge = long_segment.twin_edge.previous_edge
    long_segment.twin_edge.previous_edge.next_edge = bottom_edge_2

    # Naprawienie rekordów ścian dla shorter_edge oraz nowej bottom_edge
    mergeFaceInformationForEdges(Q, len(Q.E) - 1, longer_edge_index, precision)
    mergeFaceInformationForEdges(Q, shorter_edge_index, longer_edge_index, precision)

    # Usunięcie dłuższej krawędzi z wierzchołka w1 (zostanie tylko krótsza krawędź)
    if D.isSamePoint(Q.E[longer_edge_index].beginning, event):
        if isCloseEdge(D, vertex.incidental_edge, Q.E[longer_edge_index]):
            vertex.incidental_edge = Q.E[longer_edge_index].twin_edge.next_edge
        delEdgeFromVertex(Q.E[longer_edge_index])
    else:
        if isCloseEdge(D, vertex.incidental_edge, Q.E[longer_edge_index].twin_edge):
            vertex.incidental_edge = Q.E[longer_edge_index].twin_edge.next_edge
        delEdgeFromVertex(Q.E[longer_edge_index].twin_edge)

    # Zwraca krotkę (lewa krawędź, prawa krawędź), pomiędzy którymi należy włożyć krawędź
    insert_place = D.findPlaceForInsert(ending_point_1, bottom_edge_1)

    insert_place_inner_edge = D.getInnerHalfEdge(insert_place[0], ending_point_1)
    insert_place_inner_edge.next_edge = bottom_edge_1
    bottom_edge_1.previous_edge = insert_place_inner_edge

    insert_place_outer_edge = D.getOuterHalfEdge(insert_place[1], ending_point_1)
    insert_place_outer_edge.previous_edge = bottom_edge_2
    bottom_edge_2.next_edge = insert_place_outer_edge

    # Zmiana L w w3 dla dłuższej krawędzi (usuwamy długą, dodajemy zieloną)
    vertex_to_change = Q.searchFromPoint(Q.tree_head, ending_point_2.x, ending_point_2.y, precision)
    del vertex_to_change.L[longer_edge_index]
    vertex_to_change.L[len(Q.E)-1] = True

    # Dodanie krawędzi zielonej do U w w2, ponieważ tam będzie się ona zaczynała
    vertex_to_change = Q.searchFromPoint(Q.tree_head, ending_point_1.x, ending_point_1.y, precision)
    vertex_to_change.U.append(len(Q.E)-1)

    # Usunięcie dłuższej krawędzi z DCEL
    D.delEdge(Q.E[longer_edge_index])


# Przepisuje informację o ścianach z krawędzi 1 -> 2
def mergeFaceInformationForEdges(Q, edge_index_1, edge_index_2, precision):
    if isBeginning(Q.E, Q.E[edge_index_1].beginning, edge_index_1, precision) == 0:
        half_edge_1 = Q.E[edge_index_1]
    else:
        half_edge_1 = Q.E[edge_index_1].twin_edge

    if isBeginning(Q.E, Q.E[edge_index_2].beginning, edge_index_2, precision) == 0:
        half_edge_2 = Q.E[edge_index_2]
    else:
        half_edge_2 = Q.E[edge_index_2].twin_edge

    for i in range(len(half_edge_2.incidental_faces)):
        half_edge_1.incidental_faces.append(half_edge_2.incidental_faces[i])

    half_edge_1 = half_edge_1.twin_edge
    half_edge_2 = half_edge_2.twin_edge

    for i in range(len(half_edge_2.incidental_faces)):
        half_edge_1.incidental_faces.append(half_edge_2.incidental_faces[i])


def mergeUniqueValues(L, C):
    res = {}

    for L_index in L.keys():
        if not doesContainEdgeIndex(res, L_index):
            res[L_index] = True

    for C_index in C.keys():
        if not doesContainEdgeIndex(res, C_index):
            res[C_index] = True

    return [x for x in res.keys()]


# point1 później niż point2
def getSlope(point1, point2, precision):
    if point1[1] > point2[1]:
        point1, point2 = point2, point1
    elif isCloseVal(point1[1], point2[1], precision) and point1[0] < point2[0]:
        point1, point2 = point2, point1

    return (point2[1] - point1[1]) / (point2[0] - point1[0])


def getEdgeLength(edge):
    return np.sqrt(np.power(edge.twin_edge.beginning.x - edge.beginning.x, 2) + np.power(edge.twin_edge.beginning.y - edge.beginning.y, 2))


def isCloseEdge(D, half_edge_1, half_edge_2):
    return D.isSamePoint(half_edge_1.beginning, half_edge_2.beginning) and \
        D.isSamePoint(half_edge_1.twin_edge.beginning,
                      half_edge_2.twin_edge.beginning)


def isClosePoint(point1, point2, precision):
    return abs(point2[1] - point1[1]) < precision and abs(point2[0] - point1[0]) < precision


def isCloseVal(val1, val2, precision):
    return abs(val2 - val1) < precision


def isPointBetween(point1, point2, x, y, precision):
    if min(point1.x, point2.x) <= x <= max(point1.x, point2.x) and \
            min(point1.y, point2.y) <= y <= max(point1.y, point2.y):
        return True

    flag_x = False
    flag_y = False

    if min(point1.x, point2.x) <= x <= max(point1.x, point2.x) or \
        (isCloseVal(min(point1.x, point2.x), x, precision) and x <= max(point1.x, point2.x)) or \
            (isCloseVal(max(point1.x, point2.x), x, precision) and x >= min(point1.x, point2.x)):
        flag_x = True

    if min(point1.y, point2.y) <= y <= max(point1.y, point2.y) or \
            (isCloseVal(min(point1.y, point2.y), y, precision) and y <= max(point1.y, point2.y)) or \
            (isCloseVal(max(point1.y, point2.y), y, precision) and y >= min(point1.y, point2.y)):
        flag_y = True

    return flag_x and flag_y


def det2x2(p0, p1, p2, precision):
    if p1.y > p0.y:
        p0, p1 = p1, p0
    elif p1.y == p0.y and p1.x < p0.x:
        p0, p1 = p1, p0

    matrix = np.array([[p0.x - p2.x, p1.x - p2.x],
                       [p0.y - p2.y, p1.y - p2.y]])

    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(det) < precision:
        return 0
    elif det > 0:
        return 2
    else:
        return 1


def handleIntersectionA(D, Q, event, precision):
    intersecting_edges = list(event.C.keys())
    segment1 = Q.E[intersecting_edges[0]]
    segment2 = Q.E[intersecting_edges[1]]
    vertex = VertexRecord(event.x, event.y)

    # Stworzenie i połączenie ze sobą nowych półkrawędzi
    """1."""
    e11 = HalfEdgeRecord(beginning=vertex)
    e12 = HalfEdgeRecord(beginning=segment1.beginning)
    e11_ = HalfEdgeRecord(beginning=vertex)
    e12_ = HalfEdgeRecord(beginning=segment1.twin_edge.beginning)

    e11.twin_edge = e12
    e12.twin_edge = e11
    e11_.twin_edge = e12_
    e12_.twin_edge = e11_

    e21 = HalfEdgeRecord(beginning=vertex)
    e22 = HalfEdgeRecord(beginning=segment2.beginning)
    e21_ = HalfEdgeRecord(beginning=vertex)
    e22_ = HalfEdgeRecord(beginning=segment2.twin_edge.beginning)

    e21.twin_edge = e22
    e22.twin_edge = e21
    e21_.twin_edge = e22_
    e22_.twin_edge = e21_

    # Dodaj nowe półkrawędzie do DoubleLinkedEdgeList
    """2."""
    D.edges[((e11.beginning.x, e11.beginning.y),
             (e12.beginning.x, e12.beginning.y))] = e11
    D.edges[((e21.beginning.x, e21.beginning.y),
             (e22.beginning.x, e22.beginning.y))] = e21
    D.edges[((e11_.beginning.x, e11_.beginning.y),
             (e12_.beginning.x, e12_.beginning.y))] = e11_
    D.edges[((e21_.beginning.x, e21_.beginning.y),
             (e22_.beginning.x, e22_.beginning.y))] = e21_

    Q.E.append(e11)
    Q.E.append(e21)
    Q.E.append(e11_)
    Q.E.append(e21_)

    # Naprawienie rekordów ścian dla nowych półkrawędzi
    # segment1
    mergeFaceInformationForEdges(Q, len(Q.E)-4, intersecting_edges[0], precision)
    mergeFaceInformationForEdges(Q, len(Q.E)-2, intersecting_edges[0], precision)
    # segment2
    mergeFaceInformationForEdges(Q, len(Q.E)-3, intersecting_edges[1], precision)
    mergeFaceInformationForEdges(Q, len(Q.E)-1, intersecting_edges[1], precision)

    # Naprawianie wskaźników na końcach odcinków wejściowych
    """3."""
    e11.next_edge = segment1.twin_edge.next_edge
    segment1.twin_edge.next_edge.previous_edge = e11
    e12.previous_edge = segment1.previous_edge
    segment1.previous_edge.next_edge = e12

    e21.next_edge = segment2.twin_edge.next_edge
    segment2.twin_edge.next_edge.previous_edge = e21
    e22.previous_edge = segment2.previous_edge
    segment2.previous_edge.next_edge = e22

    e11_.next_edge = segment1.next_edge
    segment1.next_edge.previous_edge = e11_
    e12_.previous_edge = segment1.twin_edge.previous_edge
    segment1.twin_edge.previous_edge.next_edge = e12_

    e21_.next_edge = segment2.next_edge
    segment2.next_edge.previous_edge = e21_
    e22_.previous_edge = segment2.twin_edge.previous_edge
    segment2.twin_edge.previous_edge.next_edge = e22_

    # Potencjalne naprawienie krawędzi incydentalnych dla wierzchołków
    """4."""
    if isCloseEdge(D, segment1.beginning.incidental_edge, segment1):
        segment1.beginning.incidental_edge = e12

    if isCloseEdge(D, segment1.twin_edge.beginning.incidental_edge, segment1.twin_edge):
        segment1.twin_edge.beginning.incidental_edge = e12_

    if isCloseEdge(D, segment2.beginning.incidental_edge, segment2):
        segment2.beginning.incidental_edge = e22

    if isCloseEdge(D, segment2.twin_edge.beginning.incidental_edge, segment2.twin_edge):
        segment2.twin_edge.beginning.incidental_edge = e22_

    """5."""
    edge_list = [e11, e11_, e21, e21_]
    slope_list = list(map(lambda x: D.getSlope(x, vertex), edge_list))

    quarters = [[] for _ in range(4)]
    for e, (quarter, slope) in enumerate(slope_list):
        quarters[quarter-1].append((slope, edge_list[e]))

    for i in range(len(quarters)):
        quarters[i].sort(key=lambda x: x[0])

    slope_list_sorted = []
    for i in range(len(quarters)-1, -1, -1):
        slope_list_sorted += quarters[i]

    slope_list_sorted = [x[1] for x in slope_list_sorted]

    slope_list_sorted.reverse()

    for i in range(len(slope_list_sorted)):
        curr_edge_left = D.getOuterHalfEdge(slope_list_sorted[i], vertex)
        # curr_edge_right = D.getInnerHalfEdge(slope_list_sorted[i], vertex)
        curr_edge_left.previous_edge = D.getInnerHalfEdge(
            slope_list_sorted[(i - 1) % len(slope_list_sorted)], vertex)
        curr_edge_left.previous_edge.next_edge = curr_edge_left
        # curr_edge_right.next_edge = D.getOuterHalfEdge(slope_list_sorted[(i + 1) % len(slope_list_sorted)], vertex)
        # curr_edge_right.next_edge.previous_edge = curr_edge_right

    """6."""
    vertex.incidental_edge = slope_list_sorted[0]
    D.vertices[(event.x, event.y)] = vertex

    res_edge_1 = None
    res_edge_2 = None

    """7."""
    slope1 = D.getSlope(e11, vertex)
    slope2 = D.getSlope(e21, vertex)

    if slope1[0] == 2 or slope1[0] == 3 or (slope1[0] == 1 and isCloseVal(slope1[1], 0, precision)):
        res_edge_1 = len(Q.E) - 4
    else:
        res_edge_1 = len(Q.E) - 2

    if slope2[0] == 2 or slope2[0] == 3 or (slope2[0] == 1 and isCloseVal(slope1[1], 0, precision)):
        res_edge_2 = len(Q.E) - 3
    else:
        res_edge_2 = len(Q.E) - 1

    return {res_edge_1: True, res_edge_2: True}


def handleIntersectionB(D, Q, event, vertex, precision):
    segment = None
    segment_index = None
    for edge_index in event.C.keys():
        if not doesContainEdgeIndex(event.L, edge_index):
            segment = Q.E[edge_index]
            segment_index = edge_index
            break


    # nowe półkrawędzie dla e' oraz e"
    """1."""
    e11 = HalfEdgeRecord(beginning=vertex)
    e12 = HalfEdgeRecord(beginning=segment.beginning)

    """2."""
    e11.twin_edge = e12
    e12.twin_edge = e11

    e21 = HalfEdgeRecord(beginning=vertex)
    e22 = HalfEdgeRecord(beginning=segment.twin_edge.beginning)

    e21.twin_edge = e22
    e22.twin_edge = e21

    # łączymy nowe półkrawędzie z poprzednimi i następnymi krawędziami dla bazowej krawędzi 'segment'
    """3."""
    e12.previous_edge = segment.previous_edge
    segment.previous_edge.next_edge = e12
    e11.next_edge = segment.twin_edge.next_edge
    segment.twin_edge.next_edge.previous_edge = e11

    e22.previous_edge = segment.twin_edge.previous_edge
    segment.twin_edge.previous_edge.next_edge = e22
    e21.next_edge = segment.next_edge
    segment.next_edge.previous_edge = e21

    # Potencjalne naprawienie krawędzi incydentalnych dla wierzchołków
    """5."""
    if isCloseEdge(D, segment.beginning.incidental_edge, segment):
        segment.beginning.incidental_edge = e12

    if isCloseEdge(D, segment.twin_edge.beginning.incidental_edge, segment.twin_edge):
        segment.twin_edge.beginning.incidental_edge = e22

    # dodajemy nowe rekordy półkrawędzi do słownika
    """4."""
    D.edges[((e11.beginning.x, e11.beginning.y),
             (e12.beginning.x, e12.beginning.y))] = e11
    D.edges[((e21.beginning.x, e21.beginning.y),
             (e22.beginning.x, e22.beginning.y))] = e21
    # Q.E.append(((e11.beginning.x, e11.beginning.y), (e12.beginning.x, e12.beginning.y)))
    # Q.E.append(((e21.beginning.x, e21.beginning.y), (e22.beginning.x, e22.beginning.y)))
    Q.E.append(e11)
    Q.E.append(e21)

    # Naprawienie rekordów ścian dla nowych półkrawędzi
    mergeFaceInformationForEdges(Q, len(Q.E)-2, segment_index, precision)
    mergeFaceInformationForEdges(Q, len(Q.E)-1, segment_index, precision)

    slope1 = D.getSlope(e11, vertex)
    # slope2 = D.getSlope(e21, vertex)

    # start_slope = D.getSlope(vertex.incidental_edge, vertex)

    """7."""
    if slope1[0] == 2 or slope1[0] == 3 or (slope1[0] == 1 and isCloseVal(slope1[1], 0, precision)):
        res_edge = {len(Q.E) - 2: True}
    else:
        res_edge = {len(Q.E) - 1: True}

    """6."""
    # (lewa krawędź, prawa krawędź) pomiędzy którymi ma zostać włożonoa nowa półkrawędź
    left_insert = D.findPlaceForInsert(vertex, e11)

    left_insert_inner = D.getInnerHalfEdge(left_insert[0], vertex)
    left_insert_inner.next_edge = e11
    e11.previous_edge = left_insert_inner

    left_insert_outer = D.getOuterHalfEdge(left_insert[1], vertex)
    left_insert_outer.previous_edge = e12
    e12.next_edge = left_insert_outer

    right_insert = D.findPlaceForInsert(vertex, e21)

    right_insert_inner = D.getInnerHalfEdge(right_insert[0], vertex)
    right_insert_inner.next_edge = e21
    e21.previous_edge = right_insert_inner

    right_insert_outer = D.getOuterHalfEdge(right_insert[1], vertex)
    right_insert_outer.previous_edge = e22
    e22.next_edge = right_insert_outer

    return res_edge


def overlapAnyEdge(D, Q, vertex, edges_array, edge_index, precision):
    edge_slope = D.getSlope(Q.E[edge_index], vertex)[1]

    for arr_edge_index in edges_array:
        edge_slope_2 = D.getSlope(Q.E[arr_edge_index], vertex)[1]

        if isCloseVal(edge_slope, edge_slope_2, precision):
            return arr_edge_index

    return None


def findMapping(tab_of_mapped_edges, edge_index):
    while tab_of_mapped_edges[edge_index] != edge_index:
        edge_index = tab_of_mapped_edges[edge_index]
    return edge_index


# Mapuje półkrawędzie w event.C, dzieki czemu naprawia stare indeksowanie półkrawędzi
def mapEdges(tab_of_mapped_edges, event):
    list_to_iterate = list(event.C.keys()).copy()
    for key in list_to_iterate:
        mapping = findMapping(tab_of_mapped_edges, key)
        if mapping != key:
            del event.C[key]
            event.C[mapping] = True


def bentley_ottman_algorithm(S1, S2, precision):
    points_of_intersection = []

    # Wynikowa Lista Podwójnie łączonych krawędzi
    """I.1-3"""
    D = DoubleLinkedEdgeList()
    points_of_intersection += D.joinList(S1)
    points_of_intersection += D.joinList(S2)

    indexFacesForNewDCEL(S1, S2)

    points_of_intersection_after_merging = points_of_intersection.copy()

    """II.1"""
    # Kolejka Zdarzeń
    Q = AVLTree(precision)
    Q.E = list(D.edges.values())
    Q.createAVLTree()

    # Tablica zmapowanych półkrawędzi
    mapped_edges = [i for i in range(len(Q.E))]

    """II.2"""
    first_event = Q.getMinValueNode(Q.tree_head)
    Q.tree_head = Q.deleteAVL(
        Q.tree_head, None, first_event.x, first_event.y, precision)

    # Stan Zamiatania
    T = AVLTreeBroomState()

    if len(first_event.U) == 1:
        T.tree_head = StateRecord(first_event.U[0])
    elif len(first_event.U) > 1:
        T.tree_head = StateRecord(first_event.U[0])

        for i in range(1, len(first_event.U)):
            T.tree_head = T.insertAVL(
                T.tree_head, first_event.U[i], first_event, Q.E, precision)

    """II.3"""
    while not Q.hasQueueEnded():

        """II.4"""
        event = Q.getMinValueNode(Q.tree_head)
        Q.tree_head = Q.deleteAVL(
            Q.tree_head, None, event.x, event.y, precision)
        new_edges_C = {}
        """II.5"""
        mapEdges(mapped_edges, event)

        """II.6"""
        """II.7"""
        # Zapisujemy w nowej tablicy wszystkie unikalne krawędzie
        merged_L_C = mergeUniqueValues(event.L, event.C)
        # Usuń odcinki ze zbiorów L(p) i C(p)
        if len(merged_L_C) != 0:

            """II.8"""
            for _ in range(len(merged_L_C)-1):
                try:
                    T.tree_head, deleted_node_index = T.deleteAVLbyPoint(
                        T.tree_head, event, Q.E, precision)
                    merged_L_C.pop(merged_L_C.index(deleted_node_index))

                except ValueError:
                    event.C[deleted_node_index] = True
                    points_of_intersection.append((event.x, event.y))

            # Po usunięciu wszystkich odcinków oprócz jednego musimy również sprawdzić
            # czy ich poprzednik i następnik się nie przecinają
            # Możemy to sprawdzić dopiero na końcu, ponieważ usuwamy wszystkie krawędzie zawierające (event.x, event.y),
            # dlatego sprawdzamy przecięcia krawędzi na lewo i na prawo od tego punktu (ponieważ wszystkie pomiędzy
            # zostaną usunięte)

            """II.9"""
            curr_node_del = T.search(T.tree_head, merged_L_C[-1], event, Q.E, precision)
            left_node_del = T.findLeft(curr_node_del)
            right_node_del = T.findRight(curr_node_del)

            if left_node_del != None and right_node_del != None:
                intersection_del = Q.findIntersection(
                    Q.E[left_node_del.edge_index], Q.E[right_node_del.edge_index], precision)

                if intersection_del != None:
                    # Dodaj nowy punkt zdarzeń do kolejki tylko wtedy, jeżeli przecięcie występuje pod miotłą
                    # (po obecnym punkcie zdarzeń)
                    if event.y > intersection_del[1] or isCloseVal(event.y, intersection_del[1], precision) and event.x < intersection_del[0]:

                        vertex_node = Q.searchFromPoint(
                            Q.tree_head, intersection_del[0], intersection_del[1], precision)

                        # Jeżeli tablica C nie jest pusta, to znaczy, że ten punkt przecięcia został już znaleziony
                        if vertex_node == None or (vertex_node != None and len(vertex_node.C.keys()) == 0 and
                                                   (doesContainEdgeIndex(vertex_node.L, left_node_del.edge_index) 
                                                    != doesContainEdgeIndex(vertex_node.L, right_node_del.edge_index))):
                            points_of_intersection.append(intersection_del)

                        if vertex_node == None:
                            Q.tree_head = Q.insertAVL(Q.tree_head, None, [left_node_del.edge_index, right_node_del.edge_index],
                                                      intersection_del[0], intersection_del[1], 2, precision)
                        else:
                            if doesContainEdgeIndex(vertex_node.L, left_node_del.edge_index) and not \
                                    doesContainEdgeIndex(vertex_node.L, right_node_del.edge_index):
                                vertex_node.C[right_node_del.edge_index] = True

                            elif not doesContainEdgeIndex(vertex_node.L, left_node_del.edge_index) and \
                                    doesContainEdgeIndex(vertex_node.L, right_node_del.edge_index):
                                vertex_node.C[left_node_del.edge_index] = True

            # Usuń ostatni punkt
            """II.10"""
            T.tree_head, deleted_node_index = T.deleteAVLbyPoint(
                T.tree_head, event, Q.E, precision)

        else:
            """II.11"""
            # Próba usunięcia krawędzi, która przechodziłaby przez wierzchołek, a nie była w nim znaleziona
            # (przecięcie typu B)
            try:
                try_delete_node, deleted_node_index = T.deleteAVLbyPoint(
                    T.tree_head, event, Q.E, precision)
                event.C[deleted_node_index] = True

                T.tree_head = try_delete_node

                points_of_intersection.append((event.x, event.y))

            except AttributeError:
                pass

        """II.12"""
        # Handlowanie przecięcia typu A
        if len(event.C.keys()) == 2:
            new_edges_C = handleIntersectionA(D, Q, event, precision)

            edge_1 = Q.E[list(event.C.keys())[0]]
            edge_2 = Q.E[list(event.C.keys())[1]]
            D.delEdge(edge_1)
            D.delEdge(edge_2)

            if isBeginning(Q.E, edge_1.beginning, list(event.C.keys())[0], precision) == 0:
                point_for_search = edge_1.twin_edge.beginning
            else:
                point_for_search = edge_1.beginning

            vertex_to_change = Q.searchFromPoint(
                Q.tree_head, point_for_search.x, point_for_search.y, precision)
            del vertex_to_change.L[list(event.C.keys())[0]]
            vertex_to_change.L[list(new_edges_C.keys())[0]] = True

            if isBeginning(Q.E, edge_2.beginning, list(event.C.keys())[1], precision) == 0:
                point_for_search = edge_2.twin_edge.beginning
            else:
                point_for_search = edge_2.beginning

            vertex_to_change = Q.searchFromPoint(
                Q.tree_head, point_for_search.x, point_for_search.y, precision)
            del vertex_to_change.L[list(event.C.keys())[1]]
            vertex_to_change.L[list(new_edges_C.keys())[1]] = True

            # Mapowanie starych półkrawędzi
            mapped_edges += [i for i in range(len(mapped_edges),
                                              len(mapped_edges)+4)]
            mapped_edges[list(event.C.keys())[0]] = list(new_edges_C.keys())[0]
            mapped_edges[list(event.C.keys())[1]] = list(new_edges_C.keys())[1]

        elif len(event.C.keys()) == 1:
            curr_vertex = D.vertices[(event.x, event.y)]
            new_edges_C = handleIntersectionB(
                D, Q, event, curr_vertex, precision)
            edge_1 = D.getEdge(Q.E[list(event.C.keys())[0]])
            D.delEdge(edge_1)

            if isBeginning(Q.E, edge_1.beginning, list(event.C.keys())[0], precision) == 0:
                point_for_search = edge_1.twin_edge.beginning
            else:
                point_for_search = edge_1.beginning

            vertex_to_change = Q.searchFromPoint(
                Q.tree_head, point_for_search.x, point_for_search.y, precision)
            del vertex_to_change.L[list(event.C.keys())[0]]
            vertex_to_change.L[list(new_edges_C.keys())[0]] = True

            # Mapowanie starej półkrawędzi
            mapped_edges += [i for i in range(len(mapped_edges),
                                              len(mapped_edges)+2)]
            mapped_edges[list(event.C.keys())[0]] = list(new_edges_C.keys())[0]

        """II.13"""
        """II.14"""
        if len(event.U) + 2*len(event.C.keys()) > 1:
            if len(new_edges_C.keys()) != 0:
                event.C = new_edges_C

            """II.15"""
            order = findOrder(D, Q, event.U, event.L, event.C, Q.E,
                              mapped_edges, event, points_of_intersection, precision)

            """II.16"""
            # Dodaj wszystkie krawędzie w order
            for i in range(len(order)):
                T.tree_head = T.insertAVL(
                    T.tree_head, order[i], event, Q.E, precision)

            """II.17"""
            curr_node_left = T.search(T.tree_head, order[0], event, Q.E, precision)
            curr_node_right = T.search(T.tree_head, order[-1], event, Q.E, precision)

            left_node = T.findLeft(curr_node_left)
            right_node = T.findRight(curr_node_right)

            # Przecięcie z lewą stroną
            if left_node != None:
                intersection = Q.findIntersection(
                    Q.E[left_node.edge_index], Q.E[curr_node_left.edge_index], precision)
            else:
                intersection = None
            if intersection != None:
                # Dodaj nowy punkt zdarzeń do kolejki tylko wtedy, jeżeli przecięcie występuje pod miotłą
                # (po obecnym punkcie zdarzeń)
                if event.y > intersection[1] or event.y == intersection[1] and event.x < intersection[0]:

                    vertex_node = Q.searchFromPoint(
                        Q.tree_head, intersection[0], intersection[1], precision)

                    # Jeżeli tablica C nie jest pusta, to znaczy, że ten punkt przecięcia został już znaleziony
                    if vertex_node == None or (vertex_node != None and len(vertex_node.C.keys()) == 0 and
                                               (doesContainEdgeIndex(vertex_node.L, left_node.edge_index) != doesContainEdgeIndex(vertex_node.L, curr_node_left.edge_index))):
                        points_of_intersection.append(intersection)

                    if vertex_node == None:
                        Q.tree_head = Q.insertAVL(Q.tree_head, None, [left_node.edge_index, curr_node_left.edge_index],
                                                  intersection[0], intersection[1], 2, precision)
                    # Istnieją przypadki, w których więcej razy porównamy 2 odcinki
                    else:
                        if doesContainEdgeIndex(vertex_node.L, left_node.edge_index) and not \
                                doesContainEdgeIndex(vertex_node.L, curr_node_left.edge_index):
                            vertex_node.C[curr_node_left.edge_index] = True

                        elif not doesContainEdgeIndex(vertex_node.L, left_node.edge_index) and \
                                doesContainEdgeIndex(vertex_node.L, curr_node_left.edge_index):
                            vertex_node.C[left_node.edge_index] = True

                elif isCloseVal(event.y, intersection[1], precision) and isCloseVal(event.x, intersection[0], precision) \
                        and len(event.C.keys()) == 0 and len(event.L.keys()) == 0:
                    points_of_intersection.append(intersection)

                    if D.isSamePoint(Q.E[left_node.edge_index].beginning, event) or \
                            D.isSamePoint(Q.E[left_node.edge_index].twin_edge.beginning, event):
                        event.C[curr_node_left.edge_index] = True
                    else:
                        event.C[left_node.edge_index] = True

            # Przecięcie z prawą stroną
            if right_node != None:
                intersection = Q.findIntersection(Q.E[curr_node_right.edge_index], Q.E[right_node.edge_index], precision)
            else:
                intersection = None
            if intersection != None:
                if event.y > intersection[1] or event.y == intersection[1] and event.x < intersection[0]:

                    vertex_node = Q.searchFromPoint(
                        Q.tree_head, intersection[0], intersection[1], precision)

                    if vertex_node == None or (vertex_node != None and len(vertex_node.C.keys()) == 0 and
                                               (doesContainEdgeIndex(vertex_node.L, right_node.edge_index) != doesContainEdgeIndex(vertex_node.L, curr_node_right.edge_index))):
                        points_of_intersection.append(intersection)

                    if vertex_node == None:
                        Q.tree_head = Q.insertAVL(Q.tree_head, None, [curr_node_right.edge_index, right_node.edge_index],
                                                  intersection[0], intersection[1], 2, precision)
                    else:
                        if doesContainEdgeIndex(vertex_node.L, right_node.edge_index) and not \
                                doesContainEdgeIndex(vertex_node.L, curr_node_right.edge_index):
                            vertex_node.C[curr_node_right.edge_index] = True

                        elif not doesContainEdgeIndex(vertex_node.L, right_node.edge_index) and \
                                doesContainEdgeIndex(vertex_node.L, curr_node_right.edge_index):
                            vertex_node.C[right_node.edge_index] = True

                elif isCloseVal(event.y, intersection[1], precision) and isCloseVal(event.x, intersection[0], precision) \
                        and len(event.C.keys()) == 0 and len(event.L.keys()) == 0:
                    points_of_intersection.append(intersection)

                    if D.isSamePoint(Q.E[right_node.edge_index].beginning, event) or \
                            D.isSamePoint(Q.E[right_node.edge_index].twin_edge.beginning, event):
                        event.C[curr_node_right.edge_index] = True
                    else:
                        event.C[right_node.edge_index] = True

        elif len(event.U) == 1:
            """II.18"""
            """II.19"""
            T.tree_head = T.insertAVL(
                T.tree_head, event.U[0], event, Q.E, precision)

            """II.20"""
            curr_node = T.search(
                T.tree_head, event.U[0], event, Q.E, precision)
            left_node = T.findLeft(curr_node)
            right_node = T.findRight(curr_node)

            # Przecięcie z lewą stroną
            if left_node != None:
                intersection = Q.findIntersection(
                    Q.E[left_node.edge_index], Q.E[curr_node.edge_index], precision)
            else:
                intersection = None
            if intersection != None:
                if event.y > intersection[1] or event.y == intersection[1] and event.x < intersection[0]:

                    vertex_node = Q.searchFromPoint(
                        Q.tree_head, intersection[0], intersection[1], precision)

                    if vertex_node == None or (vertex_node != None and len(vertex_node.C.keys()) == 0 and
                                               (doesContainEdgeIndex(vertex_node.L, left_node.edge_index) != doesContainEdgeIndex(vertex_node.L, curr_node.edge_index))):
                        points_of_intersection.append(intersection)

                    if vertex_node == None:
                        Q.tree_head = Q.insertAVL(Q.tree_head, None, [left_node.edge_index, curr_node.edge_index],
                                                  intersection[0], intersection[1], 2, precision)
                    else:
                        if doesContainEdgeIndex(vertex_node.L, left_node.edge_index) and not \
                                doesContainEdgeIndex(vertex_node.L, curr_node.edge_index):
                            vertex_node.C[curr_node.edge_index] = True

                        elif not doesContainEdgeIndex(vertex_node.L, left_node.edge_index) and \
                                doesContainEdgeIndex(vertex_node.L, curr_node.edge_index):
                            vertex_node.C[left_node.edge_index] = True

                # Jeżeli punkt przecięcia jest tym samym punktem, w którym zaczyna się obecny odcinek oraz
                # nie został on jeszcze znaleziony - nie ma w nim żadnego przecięcia oraz nie kończy się w nim
                # żaden odcinek (bo wtedy przecinałby się z rozpoczynającym się odcinkiem), to dodaj odcinek do
                # rozwiązania.
                # (Przecięcie typu B - przecięcie istniejącego wierzchołka)
                elif isCloseVal(event.y, intersection[1], precision) and isCloseVal(event.x, intersection[0], precision) \
                        and len(event.C.keys()) == 0 and len(event.L.keys()) == 0:
                    points_of_intersection.append(intersection)

                    if D.isSamePoint(Q.E[left_node.edge_index].beginning, event) or \
                            D.isSamePoint(Q.E[left_node.edge_index].twin_edge.beginning, event):
                        event.C[curr_node.edge_index] = True
                    else:
                        event.C[left_node.edge_index] = True

            # Przecięcie z prawą stroną
            if right_node != None:
                intersection = Q.findIntersection(
                    Q.E[curr_node.edge_index], Q.E[right_node.edge_index], precision)
            else:
                intersection = None
            if intersection != None:
                if event.y > intersection[1] or isCloseVal(event.y, intersection[1], precision) and event.x < intersection[0]:

                    vertex_node = Q.searchFromPoint(
                        Q.tree_head, intersection[0], intersection[1], precision)

                    if vertex_node == None or (vertex_node != None and len(vertex_node.C.keys()) == 0 and
                                               (doesContainEdgeIndex(vertex_node.L, right_node.edge_index) != doesContainEdgeIndex(vertex_node.L, curr_node.edge_index))):
                        points_of_intersection.append(intersection)

                    if vertex_node == None:
                        Q.tree_head = Q.insertAVL(Q.tree_head, None, [curr_node.edge_index, right_node.edge_index],
                                                  intersection[0], intersection[1], 2, precision)
                    else:
                        if doesContainEdgeIndex(vertex_node.L, right_node.edge_index) and not \
                                doesContainEdgeIndex(vertex_node.L, curr_node.edge_index):
                            vertex_node.C[curr_node.edge_index] = True

                        elif not doesContainEdgeIndex(vertex_node.L, right_node.edge_index) and \
                                doesContainEdgeIndex(vertex_node.L, curr_node.edge_index):
                            vertex_node.C[right_node.edge_index] = True

                elif isCloseVal(event.y, intersection[1], precision) and isCloseVal(event.x, intersection[0], precision) \
                        and len(event.C.keys()) == 0 and len(event.L.keys()) == 0:
                    points_of_intersection.append(intersection)

                    if D.isSamePoint(Q.E[right_node.edge_index].beginning, event) or \
                            D.isSamePoint(Q.E[right_node.edge_index].twin_edge.beginning, event):
                        event.C[curr_node.edge_index] = True
                    else:
                        event.C[right_node.edge_index] = True

    """II.21"""
    return D, points_of_intersection


def makeGraph(D, G, precision):
    Q = AVLTree(precision)
    Q.E = list(D.edges.values())
    Q.createAVLTree()

    T = AVLTreeBroomState()

    while not Q.hasQueueEnded():
        event = Q.getMinValueNode(Q.tree_head)
        Q.tree_head = Q.deleteAVL(Q.tree_head, None, event.x, event.y, precision)

        # Jeżeli obecny wierzchołek jest reprezentantem cyklu zewnętrznego
        if G.isInLookup(event):
            search_edge_index = findEndingEdgeIndexForSearch(Q, event, G)

            # Przypadek skrajny - gdy jest to lewy górny wierzchołek i nie ma żadnych krawędzi z tego cyklu
            if search_edge_index == None:
                insert_edge_index = findEdgeForInsert(Q, event, G)
                T.tree_head = T.insertAVL(T.tree_head, insert_edge_index, event, Q.E, precision)
                ending_edge = T.searchWithDFSHelp(T.tree_head, insert_edge_index, event, Q.E, precision)

            else:
                ending_edge = T.searchWithDFSHelp(T.tree_head, search_edge_index, event, Q.E, precision)

            # Obiekt typu StateRecord
            left_edge = T.findLeft(ending_edge)

            if left_edge != None:

                # Obiekt typu HalfEdgeRecord
                left_proper_edge = getRightHalfEdge(Q, left_edge.edge_index, precision)

                # Utworzenie krawędzi pomiędzy cyklem będącym reprezentowanym przez wierzchołek oraz cyklem
                # dla lewego sąsiada.
                G.adjList[G.representativesLookup[(event.x, event.y)]].append(left_proper_edge.G_vertex_id)
                G.adjList[left_proper_edge.G_vertex_id].append(G.representativesLookup[(event.x, event.y)])
            else:
                G.adjList[G.representativesLookup[(event.x, event.y)]].append(-1)

            if search_edge_index == None:
                T.tree_head, deleted_node = T.deleteAVLbyPoint(T.tree_head, event, Q.E, precision)

        deletion_list = list(event.L.keys())
        for i in range(len(deletion_list)):
            T.tree_head, deleted_node_index = T.deleteAVLbyPoint(T.tree_head, event, Q.E, precision)

        if len(event.U) > 1:
            order = findOrder(D, Q, event.U, event.L, event.C, Q.E, None, event, [], precision)

            for i in range(len(order)):
                T.tree_head = T.insertAVL(T.tree_head, order[i], event, Q.E, precision)

        elif len(event.U) == 1:
            T.tree_head = T.insertAVL(T.tree_head, event.U[0], event, Q.E, precision)


# Szukamy indeksu krawędzi kończącej się w danym punkcie, która należy do cyklu, który jest cyklem zewnętrznym i
# obecny wierzchołek jest jego reprezentantem.
# Funkcja jest wywołwana z założeniem, że obecny punkt jest w G.representativeLookup oraz cykl, który reprezentuje
# jest cyklem zewnętrznym.
def findEndingEdgeIndexForSearch(Q, event, G):
    edge_list = list(event.L.keys())

    searched_G_vertex_id = G.representativesLookup[(event.x, event.y)]

    for edge_index in edge_list:
        if Q.E[edge_index].G_vertex_id == searched_G_vertex_id or Q.E[edge_index].twin_edge.G_vertex_id == searched_G_vertex_id:
            return edge_index

    return None


def findEdgeForInsert(Q, event, G):
    searched_G_vertex_id = G.representativesLookup[(event.x, event.y)]
    for edge_index in event.U:
        if Q.E[edge_index].G_vertex_id == searched_G_vertex_id or Q.E[edge_index].twin_edge.G_vertex_id == searched_G_vertex_id:
            return edge_index

    return None


def getRightHalfEdge(Q, edge_index, precision):
    if isBeginning(Q.E, Q.E[edge_index].beginning, edge_index, precision) == 0:
        return Q.E[edge_index]
    else:
        return Q.E[edge_index].twin_edge


def fixFacesRecords(D, G):
    next_face_id = 0

    for vertex in range(G.V):
        # Jeśli jest to cykl wewnętrzny
        if G.vertexType[vertex] == 0:
            new_face = FaceRecord(next_face_id, G.edgeInCycle[vertex], None, None)
            next_face_id += 1

            # Wskaźnik do krawędzi na zewnętrznym cyklu
            new_face.edge_to_outer_cycle = G.getEdgeToOuterCycle(vertex)

            # Wskaźniki do krawędzi na dziurach
            inner_edges = {}
            G.getEdgesArrToInnerCycles(vertex, inner_edges)
            new_face.inner_edges = list(inner_edges.keys()).copy()

            # Zebranie danych ze ścian ze zbiorów wejściowych do słownika 'data_arr'
            data_arr = [None, None]
            starting_edge = G.edgeInCycle[vertex]
            for face in starting_edge.incidental_faces:
                if data_arr[face.DCEL_id-1] == None:
                    data_arr[face.DCEL_id-1] = face.data

            curr_edge = starting_edge.next_edge
            while curr_edge != starting_edge:
                for face in starting_edge.incidental_faces:
                    if data_arr[face.DCEL_id - 1] == None:
                        data_arr[face.DCEL_id - 1] = face.data
                curr_edge = curr_edge.next_edge

            # Zmergowanie danych do nowego rekordu ściany
            new_data = []
            for face_data in data_arr:
                new_data.append(face_data)

            new_face.data = new_data.copy()

            # Dodanie nowej ściany do DCEL
            D.faces.append(new_face)

            # Ustawienie rekordów ścian dla odpowiednich cykli
            G.fillCycleWithFace(new_face, vertex)

    # Jeżeli brakuje którejś informacji, próbujemy ją uzupełnić
    for face in D.faces:
        if face.edge_to_outer_cycle.beginning.x == None and face.edge_to_outer_cycle.beginning.y == None:
            continue

        if face.data[0] == None:
            visited = [False for _ in range(G.V)]
            face.data[0] = G.DFSSearchGetDataFromOuterCycle(face.edge_to_outer_cycle.G_vertex_id, 1, visited)

        elif face.data[1] == None:
            visited = [False for _ in range(G.V)]
            face.data[1] = G.DFSSearchGetDataFromOuterCycle(face.edge_to_outer_cycle.G_vertex_id, 2, visited)


def indexFacesForNewDCEL(D1, D2):
    for face in D1.faces:
        face.DCEL_id = 1

    for face in D2.faces:
        face.DCEL_id = 2


def doesContainPoint(arr, point, precision):
    for i in range(len(arr)):
        if isClosePoint(arr[i], point, precision):
            return True
    return False


# Jeżeli nie zawiera punktu, to rzuci wyjątek KeyError i zwróci False; jeżeli przejdzie, zwróci True
def doesContainEdgeIndex(dictionary, edge_index):
    try:
        dictionary[edge_index]
        return True
    except KeyError:
        return False

