import numpy as np
import json as js
import sys
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.widgets import Button
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

TOLERANCE = 0.15

SAVE_ON_SHOW = False

if SAVE_ON_SHOW:
    file_name_cnt = 0

def dist(point1, point2):
    return np.sqrt(np.power(point1[0] - point2[0], 2) + np.power(point1[1] - point2[1], 2))


class _Button_callback(object):
    def __init__(self, scenes):
        self.curr_point = None
        self.color = iter(plt.cm.rainbow(np.linspace(0, 1, 5)))
        self.map_color = next(self.color)
        self.i = 0
        self.scenes = scenes
        self.adding_points = False
        self.added_points = []
        self.adding_lines = False
        self.added_lines = []
        self.adding_maps = False
        self.added_maps = []

    def set_axes(self, ax):
        self.ax = ax

    def next_scene(self, event):
        self.i = (self.i + 1) % len(self.scenes)
        title = self.scenes[self.i].title
        self.draw(autoscaling=True, title=title)

    def prev_scene(self, event):
        self.i = (self.i - 1) % len(self.scenes)
        title = self.scenes[self.i].title
        self.draw(autoscaling=True, title=title)

    def add_point(self, event):
        self.adding_points = not self.adding_points
        self.new_line_point = None
        if self.adding_points:
            self.adding_lines = False
            self.adding_maps = False
            self.added_points.append(PointsCollection([]))

    def add_line(self, event):
        self.adding_lines = not self.adding_lines
        self.new_line_point = None
        if self.adding_lines:
            self.adding_points = False
            self.adding_maps = False
            self.added_lines.append(LinesCollection([]))

    def add_map(self, event):
        self.adding_maps = True
        self.new_line_point = None
        if self.adding_maps:
            self.map_color = next(self.color)
            self.adding_points = False
            self.adding_lines = False
            self.curr_point = None
            self.new_map()

    def new_map(self):
        self.added_maps.append(LinesCollection([], colors=self.map_color))
        self.map_points = []

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        new_point = (event.xdata, event.ydata)
        if self.adding_points:
            self.added_points[-1].add_points([new_point])
            self.draw(autoscaling=False)
        elif self.adding_lines:
            if self.new_line_point is not None:
                self.added_lines[-1].add([self.new_line_point, new_point])
                self.new_line_point = None
                self.draw(autoscaling=False)
            else:
                self.new_line_point = new_point
        elif self.adding_maps:
            if len(self.map_points) == 0:
                self.map_points.append(new_point)
                self.curr_point = new_point
            elif len(self.map_points) == 1:
                self.added_maps[-1].add([self.map_points[-1], new_point])
                self.map_points.append(new_point)
                self.curr_point = new_point
                self.draw(autoscaling=False)
            elif len(self.map_points) > 1:
                if event.button is MouseButton.RIGHT:
                    for point in self.map_points:
                        if dist(point, new_point) < (np.mean([self.ax.get_xlim(), self.ax.get_ylim()]) * TOLERANCE):
                            self.curr_point = point
                            break
                else:
                    for point in self.map_points:
                        if dist(point, new_point) < (np.mean([self.ax.get_xlim(), self.ax.get_ylim()]) * TOLERANCE):
                            self.added_maps[-1].add([self.curr_point, point])
                            self.curr_point = point
                            break
                    else:
                        self.added_maps[-1].add([self.curr_point, new_point])
                        self.map_points.append(new_point)
                        self.curr_point = new_point
                    self.draw(autoscaling=False)

    def draw(self, autoscaling=True, title=""):
        global file_name_cnt
        if not autoscaling:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        self.ax.clear()
        for collection in (self.scenes[self.i].points + self.added_points):
            if len(collection.points) > 0:
                self.ax.scatter(*zip(*(np.array(collection.points))), **collection.kwargs)
        for collection in (self.scenes[self.i].lines + self.scenes[self.i].polys + self.added_lines + self.added_maps):
            self.ax.add_collection(collection.get_collection())
        self.ax.autoscale(autoscaling)
        if not autoscaling:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        plt.title(title)
        if SAVE_ON_SHOW:
            plt.savefig(str(file_name_cnt)+'.png')
            file_name_cnt += 1
        plt.draw()


class Scene:
    def __init__(self, points=[], lines=[], polys=[], title=""):
        self.points = points
        self.lines = lines
        self.polys = polys
        self.title = title


class PointsCollection:
    def __init__(self, points, **kwargs):
        self.points = points
        self.kwargs = kwargs

    def add_points(self, points):
        self.points = self.points + points


class LinesCollection:
    def __init__(self, lines, **kwargs):
        self.lines = lines
        self.kwargs = kwargs

    def add(self, line):
        self.lines.append(line)

    def get_collection(self):
        return mcoll.LineCollection(self.lines, **self.kwargs)


# Klasa PolygonsCollection przechowuje zbiór wielokątów, które są reprezentowane przez
# tablice z kolejnymi punktami tego wielokąta.
class PolygonsCollection:
    def __init__(self, polygons, **kwargs):
        self.polygons = polygons
        self.kwargs = kwargs

    def add(self, poly):
        self.polygons.appends(poly)

    def get_collection(self):
        return mcoll.PolyCollection(self.polygons, **self.kwargs)


class Plot:
    def __init__(self, scenes=[Scene()], points=[], lines=[], polys=[], json=None, title=''):
        self.title = title
        if json is None:
            self.scenes = scenes
            if points or lines:
                self.scenes[0].points = points
                self.scenes[0].lines = lines
                self.scenes[0].polys = polys
        else:
            self.scenes = [Scene([PointsCollection(pointsCol) for pointsCol in scene["points"]],
                                 [LinesCollection(linesCol) for linesCol in scene["lines"]],
                                 [PolygonsCollection(polysCol) for polysCol in scene["polys"]])
                           for scene in js.loads(json)]

    def __configure_buttons(self):
        plt.subplots_adjust(bottom=0.2)
        ax_prev = plt.axes([0.383, 0.05, 0.253, 0.075])
        ax_next = plt.axes([0.646, 0.05, 0.253, 0.075])
        # ax_add_point = plt.axes([0.44, 0.05, 0.15, 0.075])
        # ax_add_line = plt.axes([0.28, 0.05, 0.15, 0.075])
        ax_add_map = plt.axes([0.12, 0.05, 0.253, 0.075])
        b_next = Button(ax_next, 'Next scene')
        b_next.on_clicked(self.callback.next_scene)
        b_prev = Button(ax_prev, 'Prev scene')
        b_prev.on_clicked(self.callback.prev_scene)
        # b_add_point = Button(ax_add_point, 'Add point')
        # b_add_point.on_clicked(self.callback.add_point)
        # b_add_line = Button(ax_add_line, 'Add line')
        # b_add_line.on_clicked(self.callback.add_line)
        b_add_map = Button(ax_add_map, 'Add map')
        b_add_map.on_clicked(self.callback.add_map)
        return [b_prev, b_next, b_add_map]
        # return [b_prev, b_next, b_add_point, b_add_line, b_add_map]

    def add_scene(self, scene):
        self.scenes.append(scene)

    def add_scenes(self, scenes):
        self.scenes = self.scenes + scenes

    def toJson(self):
        return js.dumps([{"points": [np.array(pointCol.points).tolist() for pointCol in scene.points],
                          "lines": [linesCol.lines for linesCol in scene.lines],
                          "polys": [polysCol.polys for polysCol in scene.polys]}
                         for scene in self.scenes])

    def get_added_points(self):
        if self.callback:
            return self.callback.added_points
        else:
            return None

    def get_added_lines(self):
        if self.callback:
            return self.callback.added_lines
        else:
            return None

    def get_added_figure(self):
        if self.callback:
            return self.callback.added_maps
        else:
            return None

    def get_added_maps(self):
        if self.callback:
            figures = self.get_added_figure()
            maps = []
            for linesCollection in figures:
                maps.append(linesCollection.lines)
            return maps
        else:
            return None

    def get_added_elements(self):
        if self.callback:
            return Scene(self.callback.added_points, self.callback.added_lines + self.callback.added_maps)
        else:
            return None

    def draw(self, buttonsoff=False):
        plt.close()
        fig = plt.figure()
        self.callback = _Button_callback(self.scenes)
        if not buttonsoff:
            self.widgets = self.__configure_buttons()
        ax = plt.axes(autoscale_on=False)
        self.callback.set_axes(ax)
        fig.canvas.mpl_connect('button_press_event', self.callback.on_click)
        if buttonsoff:
            self.callback.next_scene(None)
        if self.title != '':
            title = self.title
        else:
            title = self.callback.scenes[0].title
        self.callback.draw(title=title)
        plt.show()


def dfs_face_edges(half_edge, start_v, visited):
    visited.add(half_edge)
    if half_edge.next_edge not in visited:
        return [list(half_edge)] + dfs_face_edges(half_edge.next_edge, start_v, visited)
    return [list(half_edge)]


def find_polygons(D):
    faces = []
    visited = set()
    for edge in D.edges.values():
        # edge = v.incidental_edge
        lines = dfs_face_edges(edge, edge.beginning, visited)
        if len(lines) > 2:
            faces.append(lines)

    faces.sort(reverse=True, key=lambda l: poly_area_tuple(polygons_as_x_y(l)))
    return faces


def polygons_as_x_y(lines):
    x = []
    y = []
    for (a, _) in lines:
        x.append(a[0])
        y.append(a[1])
    return np.array(x), np.array(y)


def poly_area_tuple(t):
    return poly_area(t[0], t[1])


def poly_area(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def show_faces_as_polygons(faces_as_lines):
    fig, ax = plt.subplots()
    patches = []
    face_lines = []
    for lines in faces_as_lines:
        pol_x, pol_y = polygons_as_x_y(lines)
        pol_x = list(pol_x)
        pol_y = list(pol_y)
        pol_x.append(pol_x[0])
        pol_y.append(pol_y[0])
        face_lines.append((pol_x, pol_y))
        points = np.ones((len(lines), 2))
        for i, (a, b) in enumerate(lines):
            points[i, 0] = a[0]
            points[i, 1] = a[1]
        polygon = Polygon(points, True)
        patches.append(polygon)

    # p = PatchCollection(patches, cmap=matplotlib.cm.jet)
    p = PatchCollection(patches, cmap=get_cmap(len(faces_as_lines)))
    colors = 100 * np.random.rand(len(faces_as_lines))
    p.set_array(np.array(colors))
    ax.add_collection(p)
    plt.title(f"{len(patches)-1} colored faces")

    for (x, y) in face_lines:
        plt.plot(x, y, color='black')

    plt.show()


# wyznaczenie offsetu dla półkrawędzi (aby je rozdzielić) używając odległości pomiędzy dwoma prostymi:
# d = |C1 - C2| / sqrt(A^2 + B^2)
# Znamy współ. A, B oraz C1, chcemy wyznaczyć C2 tak, żeby nowa prosta była po lewej stronie starej,
# w przyjętej orientacji. d przyjmujemy jako pewną wartość (należy pamiętać, że ostateczna odległość
# będzie 2 razy większa)
# To, czy górny wyraz będzie C1 - C2, czy C2 - C1 oznacza czy prosta 2 będzie nad prostą 1, czy na odwrót.
# Zależnie od zorientowania wejściowej półkrawędzi:
# - 1 ćwiartka - C2 góra
# - 2 ćwiartka - C2 góra
# - 3 ćwiartka - C2 dół
# - 4 ćwiartka - C2 dół
# więc wnioskujemy, że to która półkrawędź będzie na górze/dole wynika z delta_x
def calcOffset(half_edge):
    delta_y = half_edge.twin_edge.beginning.y - half_edge.beginning.y
    delta_x = half_edge.twin_edge.beginning.x - half_edge.beginning.x

    if delta_x == 0:
        a = sys.maxsize
    else:
        a = delta_y / delta_x

    # b = half_edge.beginning.y - a*half_edge.beginning.x
    if a == sys.maxsize:
        a2 = 0
    elif a == 0:
        a2 = sys.maxsize
    else:
        a2 = -1/a

    new_delta_y = a2
    new_delta_x = 1

    if new_delta_y == sys.maxsize:
        new_delta_y = 1
        new_delta_x = 0

    else:
        vector_length = np.sqrt(np.power(new_delta_x, 2) + np.power(new_delta_y, 2))

        new_delta_y /= vector_length
        new_delta_x /= vector_length

    if (delta_y > 0 and delta_x > 0) or (delta_y >= 0 and delta_x < 0):
        new_delta_y = -new_delta_y
        new_delta_x = -new_delta_x

    new_delta_x *= 0.05
    new_delta_y *= 0.05

    return (new_delta_x, new_delta_y)


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def visualize(D, G):
    polygons = []
    for face in D.faces:
        if G.vertexType[face.outer_edge.G_vertex_id] == 0:
            points = []
            starting_edge = face.outer_edge
            points.append((starting_edge.beginning.x, starting_edge.beginning.y))
            curr_edge = starting_edge.next_edge

            while curr_edge != starting_edge:
                points.append((curr_edge.beginning.x, curr_edge.beginning.y))
                curr_edge = curr_edge.next_edge

            if face.data[0] != None:
                poly_blue = 1.0*face.data[0]/100
            else:
                poly_blue = 0.0

            if face.data[1] != None:
                poly_green = 1.0*face.data[1]/100
            else:
                poly_green = 0.0

            poly_color_1 = (0.0, 0.0, poly_blue, poly_blue)
            poly_color_2 = (0.0, poly_green, 0.0, poly_green)
            poly_color_fill = (1.0, 1.0, 1.0, 0.0)

            polygons.append(PolygonsCollection(polygons=[points], color=poly_color_fill, edgecolor=poly_color_1, hatch="//"))
            polygons.append(PolygonsCollection(polygons=[points], color=poly_color_fill, edgecolor=poly_color_2, hatch="\\"))

    return Scene(polys=polygons, title='')