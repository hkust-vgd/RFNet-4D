from scipy.spatial import Delaunay
from itertools import combinations
import numpy as np
from im2mesh.utils import voxels


class MultiGridExtractor(object):
    def __init__(self, resolution0, threshold):
        # Attributes
        self.resolution = resolution0
        self.threshold = threshold

        # Voxels are active or inactive,
        # values live on the space between voxels and are either
        # known exactly or guessed by interpolation (unknown)
        shape_voxels = (resolution0, ) * 3
        shape_values = (resolution0 + 1, ) * 3
        self.values = np.empty(shape_values)
        self.value_known = np.full(shape_values, False)
        self.voxel_active = np.full(shape_voxels, True)

    def query(self):
        """
        > The function `query` returns the indices of all the locations in the grid that are active but
        unknown
        :return: The points that are active but unknown.
        """
        # Query locations in grid that are active but unkown
        idx1, idx2, idx3 = np.where(~self.value_known & self.value_active)
        points = np.stack([idx1, idx2, idx3], axis=-1)
        return points

    def update(self, points, values):
        """
        It updates the values of the voxels at the given points, and updates the activity status of the
        voxels
        
        :param points: a list of points to update
        :param values: the values of the voxels
        """
        # Update locations and set known status to true
        idx0, idx1, idx2 = points.transpose()
        self.values[idx0, idx1, idx2] = values
        self.value_known[idx0, idx1, idx2] = True

        # Update activity status of voxels accordings to new values
        self.voxel_active = ~self.voxel_empty

    def increase_resolution(self):
        """
        It takes a 3D array of values and a 3D array of booleans indicating which values are known, and
        returns a new 3D array of values and a new 3D array of booleans, where the new 3D array of
        values is twice as large in each dimension, and the new 3D array of booleans is True for every
        other value in the new 3D array of values
        """
        self.resolution = 2 * self.resolution
        shape_values = (self.resolution + 1, ) * 3

        value_known = np.full(shape_values, False)
        value_known[::2, ::2, ::2] = self.value_known
        values = upsample3d_nn(self.values)
        values = values[:-1, :-1, :-1]

        self.values = values
        self.value_known = value_known
        self.voxel_active = upsample3d_nn(self.voxel_active)

    @property
    def occupancies(self):
        """
        It returns a boolean array of the same shape as the input array, where each element is True if
        the corresponding element in the input array is less than the threshold, and False otherwise
        :return: The values that are less than the threshold.
        """
        return (self.values < self.threshold)

    @property
    def value_active(self):
        """
        "If a voxel is active, then all of its adjacent values are active."
        
        The function is a little more complicated than that, but that's the gist of it
        :return: A boolean array of the same shape as the values array.
        """
        value_active = np.full(self.values.shape, False)

        # Active if adjacent to active voxel
        value_active[:-1, :-1, :-1] |= self.voxel_active
        value_active[:-1, :-1, 1:] |= self.voxel_active
        value_active[:-1, 1:, :-1] |= self.voxel_active
        value_active[:-1, 1:, 1:] |= self.voxel_active
        value_active[1:, :-1, :-1] |= self.voxel_active
        value_active[1:, :-1, 1:] |= self.voxel_active
        value_active[1:, 1:, :-1] |= self.voxel_active
        value_active[1:, 1:, 1:] |= self.voxel_active

        return value_active

    @property
    def voxel_known(self):
        """
        This function checks if the voxel is occupied or not
        :return: The voxel_known function is returning the value of the voxel_known variable.
        """
        value_known = self.value_known
        voxel_known = voxels.check_voxel_occupied(value_known)
        return voxel_known

    @property
    def voxel_empty(self):
        """
        > If the voxel is not empty, return True, otherwise return False
        :return: The occupancy grid.
        """
        occ = self.occupancies
        return ~voxels.check_voxel_boundary(occ)


def upsample3d_nn(x):
    """
    It takes a 3D array and returns a new 3D array that is twice as large in each dimension, with the
    values in the new array being the same as the values in the original array
    
    :param x: the input tensor
    :return: a 3D array of the same shape as the input array, but with each element repeated 8 times.
    """
    xshape = x.shape
    yshape = (2 * xshape[0], 2 * xshape[1], 2 * xshape[2])

    y = np.zeros(yshape, dtype=x.dtype)
    y[::2, ::2, ::2] = x
    y[::2, ::2, 1::2] = x
    y[::2, 1::2, ::2] = x
    y[::2, 1::2, 1::2] = x
    y[1::2, ::2, ::2] = x
    y[1::2, ::2, 1::2] = x
    y[1::2, 1::2, ::2] = x
    y[1::2, 1::2, 1::2] = x

    return y


class DelauneyMeshExtractor(object):
    """Algorithm for extacting meshes from implicit function using
    delauney triangulation and random sampling."""
    def __init__(self, points, values, threshold=0.):
        self.points = points
        self.values = values
        self.delaunay = Delaunay(self.points)
        self.threshold = threshold

    def update(self, points, values, reduce_to_active=True):
        """
        > The function takes in a set of points and values, and updates the current set of points and
        values with the new set. 
        
        The function is called in the following way:
        
        :param points: the points to be added to the triangulation
        :param values: The values of the points in the grid
        :param reduce_to_active: If True, the active simplices are found and only those points are kept,
        defaults to True (optional)
        """
        # Find all active points
        if reduce_to_active:
            active_simplices = self.active_simplices()
            active_point_idx = np.unique(active_simplices.flatten())
            self.points = self.points[active_point_idx]
            self.values = self.values[active_point_idx]

        self.points = np.concatenate([self.points, points], axis=0)
        self.values = np.concatenate([self.values, values], axis=0)
        self.delaunay = Delaunay(self.points)

    def extract_mesh(self):
        """
        It takes a list of points and values, and returns a list of vertices and triangles that form a mesh
        :return: vertices and triangles
        """
        threshold = self.threshold
        vertices = []
        triangles = []
        vertex_dict = dict()

        active_simplices = self.active_simplices()
        active_simplices.sort(axis=1)
        for simplex in active_simplices:
            new_vertices = []
            for i1, i2 in combinations(simplex, 2):
                assert (i1 < i2)
                v1 = self.values[i1]
                v2 = self.values[i2]
                if (v1 < threshold) ^ (v2 < threshold):
                    # Subdivide edge
                    vertex_idx = vertex_dict.get((i1, i2), len(vertices))
                    vertex_idx = len(vertices)
                    if vertex_idx == len(vertices):
                        tau = (threshold - v1) / (v2 - v1)
                        assert (0 <= tau <= 1)
                        p = (1 - tau) * self.points[i1] + tau * self.points[i2]
                        vertices.append(p)
                        vertex_dict[i1, i2] = vertex_idx
                    new_vertices.append(vertex_idx)

            assert (len(new_vertices) in (3, 4))
            p0 = self.points[simplex[0]]
            v0 = self.values[simplex[0]]
            if len(new_vertices) == 3:
                i1, i2, i3 = new_vertices
                p1, p2, p3 = vertices[i1], vertices[i2], vertices[i3]
                vol = get_tetrahedon_volume(np.asarray([p0, p1, p2, p3]))
                if vol * (v0 - threshold) <= 0:
                    triangles.append((i1, i2, i3))
                else:
                    triangles.append((i1, i3, i2))
            elif len(new_vertices) == 4:
                i1, i2, i3, i4 = new_vertices
                p1, p2, p3, p4 = \
                    vertices[i1], vertices[i2], vertices[i3], vertices[i4]
                vol = get_tetrahedon_volume(np.asarray([p0, p1, p2, p3]))
                if vol * (v0 - threshold) <= 0:
                    triangles.append((i1, i2, i3))
                else:
                    triangles.append((i1, i3, i2))

                vol = get_tetrahedon_volume(np.asarray([p0, p2, p3, p4]))
                if vol * (v0 - threshold) <= 0:
                    triangles.append((i2, i3, i4))
                else:
                    triangles.append((i2, i4, i3))

        vertices = np.asarray(vertices, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.int32)

        return vertices, triangles

    def query(self, size):
        """
        > We sample points from the tetrahedra formed by the active simplices of the Delaunay
        triangulation
        
        :param size: the number of points to sample
        :return: The new points are being returned.
        """
        active_simplices = self.active_simplices()
        active_simplices_points = self.points[active_simplices]
        new_points = sample_tetraheda(active_simplices_points, size=size)

        return new_points

    def active_simplices(self):
        """
        It returns the indices of the simplices that are active, i.e. the simplices that have at least
        one vertex with a value above the threshold and at least one vertex with a value below the
        threshold
        :return: The simplices that are active.
        """
        occ = (self.values >= self.threshold)
        simplices = self.delaunay.simplices
        simplices_occ = occ[simplices]

        active = (np.any(simplices_occ, axis=1)
                  & np.any(~simplices_occ, axis=1))

        simplices = self.delaunay.simplices[active]

        return simplices


def sample_tetraheda(tetraheda_points, size):
    """
    We sample a tetrahedron from the list of tetrahedra with probability proportional to its volume,
    then we sample a point from the tetrahedron with probability proportional to the volume of the
    tetrahedron
    
    :param tetraheda_points: the 4 points of the tetrahedron
    :param size: the number of points to sample
    :return: a set of points that are sampled from the tetraheda.
    """
    N_tetraheda = tetraheda_points.shape[0]
    volume = np.abs(get_tetrahedon_volume(tetraheda_points))
    probs = volume / volume.sum()

    tetraheda_rnd = np.random.choice(range(N_tetraheda), p=probs, size=size)
    tetraheda_rnd_points = tetraheda_points[tetraheda_rnd]
    weights_rnd = np.random.dirichlet([1, 1, 1, 1], size=size)
    weights_rnd = weights_rnd.reshape(size, 4, 1)
    points_rnd = (weights_rnd * tetraheda_rnd_points).sum(axis=1)

    return points_rnd


def get_tetrahedon_volume(points):
    """
    It computes the volume of a tetrahedron by computing the determinant of the matrix whose columns are
    the vectors from the fourth vertex to the other three
    
    :param points: a (batch_size, 4, 3) tensor of points
    :return: The volume of the tetrahedron.
    """
    vectors = points[..., :3, :] - points[..., 3:, :]
    volume = 1 / 6 * np.linalg.det(vectors)

    return volume
