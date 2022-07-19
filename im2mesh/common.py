# import multiprocessing
import torch
from pykdtree.kdtree import KDTree
import numpy as np
import trimesh
import glob
import os


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def chamfer_distance(points1, points2, use_kdtree=True, give_id=False):
    ''' Returns the chamfer distance for the sets of points.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        use_kdtree (bool): whether to use a kdtree
        give_id (bool): whether to return the IDs of nearest points
    '''
    if use_kdtree:
        return chamfer_distance_kdtree(points1, points2, give_id=give_id)
    else:
        return chamfer_distance_naive(points1, points2)


def chamfer_distance_naive(points1, points2):
    ''' Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set    
    '''
    assert (points1.size() == points2.size())
    batch_size, T, _ = points1.size()

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    distances = (points1 - points2).pow(2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer


def chamfer_distance_kdtree(points1, points2, give_id=False):
    ''' KD-tree based implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        give_id (bool): whether to return the IDs of the nearest points
    '''
    # Points have size batch_size x T x 3
    batch_size = points1.size(0)

    # First convert points to numpy
    points1_np = points1.detach().cpu().numpy()
    points2_np = points2.detach().cpu().numpy()

    # Get list of nearest neighbors indices
    idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
    idx_nn_12 = torch.LongTensor(idx_nn_12).to(points1.device)
    # Expands it as batch_size x 1 x 3
    idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)

    # Get list of nearest neighbors indices
    idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
    idx_nn_21 = torch.LongTensor(idx_nn_21).to(points1.device)

    # Expands it as batch_size x T x 3
    idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)

    # Compute nearest neighbors in points2 to points in points1
    # points_12[i, j, k] = points2[i, idx_nn_12_expand[i, j, k], k]
    points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)

    # Compute nearest neighbors in points1 to points in points2
    # points_21[i, j, k] = points2[i, idx_nn_21_expand[i, j, k], k]
    points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)

    # Compute chamfer distance
    chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
    chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)

    # Take sum
    chamfer = chamfer1 + chamfer2

    # If required, also return nearest neighbors
    if give_id:
        return chamfer1, chamfer2, idx_nn_12, idx_nn_21

    return chamfer


def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.

    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def transform_points(points, transform):
    ''' Transforms points with regard to passed camera information.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    assert (points.size(2) == 3)
    assert (transform.size(1) == 3)
    assert (points.size(0) == transform.size(0))

    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points @ R.transpose(1, 2) + t.transpose(1, 2)
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ K.transpose(1, 2)

    return points_out


def project_to_camera(points, transform):
    ''' Projects points to the camera plane.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    p_camera = transform_points(points, transform)
    p_camera = p_camera[..., :2] / p_camera[..., 2:]
    return p_camera


def load_and_scale_mesh(mesh_path, loc=None, scale=None):
    ''' Loads and scales a mesh.

    Args:
        mesh_path (str): mesh path
        loc (tuple): location
        scale (float): scaling factor
    '''
    mesh = trimesh.load(mesh_path, process=False)

    # Compute location and scale
    if loc is None or scale is None:
        bbox = mesh.bounding_box.bounds
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-loc)
    mesh.apply_scale(1 / scale)

    return loc, scale, mesh


def load_and_scale_mesh_sequence(mesh_dir,
                                 start_idx,
                                 loc=None,
                                 scale=None,
                                 file_ext='obj',
                                 seq_len=17,
                                 multi_interval=1):
    """
    It loads a sequence of meshes from a directory, centers them around the origin, and scales them to
    fit in a unit cube
    
    :param mesh_dir: the directory where the mesh files are stored
    :param start_idx: the index of the first frame in the sequence
    :param loc: the center of the bounding box of the first mesh in the sequence
    :param scale: the scale of the mesh
    :param file_ext: the file extension of the mesh files, defaults to obj (optional)
    :param seq_len: the number of frames in the sequence, defaults to 17 (optional)
    :param multi_interval: if you want to skip frames, set this to a number greater than 1, defaults to
    1 (optional)
    :return: loc, scale, np.stack(vertices)
    """
    mesh_files = glob.glob(os.path.join(mesh_dir, '*.%s' % file_ext))
    mesh_files.sort()
    if multi_interval > 1:
        select_mesh_files = []
        idx = start_idx
        for i in range(seq_len):
            select_mesh_files.append(mesh_files[start_idx +
                                                i * multi_interval])
        mesh_files = select_mesh_files
    else:
        mesh_files = mesh_files[start_idx:start_idx + seq_len]

    if loc is None or scale is None:
        mesh_0 = trimesh.load(mesh_files[0], process=False)
        bbox = mesh_0.bounding_box.bounds
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max()

    vertices = []
    for mesh_p in mesh_files:
        mesh = trimesh.load(mesh_p, process=False)
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)
        vertices.append(np.array(mesh.vertices, dtype=np.float32))

    return loc, scale, np.stack(vertices)


def load_and_scale_pointcloud_sequence(pointcloud_dir,
                                       start_idx,
                                       loc=None,
                                       scale=None,
                                       file_ext='npz',
                                       num_points=2048,
                                       seq_len=17,
                                       multi_interval=1):
    """
    It loads a sequence of pointclouds, and scales them to a canonical coordinate system
    
    :param pointcloud_dir: the directory where the pointclouds are stored
    :param start_idx: the index of the first pointcloud in the sequence
    :param loc: the location of the point cloud in the world coordinate system
    :param scale: the scale of the pointcloud
    :param file_ext: the file extension of the pointcloud files, defaults to npz (optional)
    :param num_points: the number of points in the pointcloud, defaults to 2048 (optional)
    :param seq_len: the number of frames in the sequence, defaults to 17 (optional)
    :param multi_interval: the number of frames to skip between each frame in the sequence, defaults to
    1 (optional)
    :return: loc0, scale0, np.stack(pointclouds)
    """
    pointcloud_files = glob.glob(
        os.path.join(pointcloud_dir, '*.%s' % file_ext))
    pointcloud_files.sort()
    if multi_interval > 1:
        select_pointcloud_files = []
        idx = start_idx
        for i in range(seq_len):
            select_pointcloud_files.append(
                pointcloud_files[start_idx + i * multi_interval])
        pointcloud_files = select_pointcloud_files
    else:
        pointcloud_files = pointcloud_files[start_idx:start_idx + seq_len]

    if loc is None or scale is None:
        pointcloud_dict = np.load(pointcloud_files[0])
        points_0 = pointcloud_dict['points'].astype(np.float32)
        loc0 = pointcloud_dict['loc'].astype(np.float32)
        scale0 = pointcloud_dict['scale'].astype(np.float32)

    pointclouds = []
    for p_file in pointcloud_files:
        pointcloud_dict = np.load(p_file)
        points = pointcloud_dict['points'].astype(np.float32)
        loc = pointcloud_dict['loc'].astype(np.float32)
        scale = pointcloud_dict['scale'].astype(np.float32)
        points = (loc + scale * points - loc0) / scale0
        pointclouds.append(points[:num_points, :])

    #return loc0, scale0, data
    return loc0, scale0, np.stack(pointclouds)
