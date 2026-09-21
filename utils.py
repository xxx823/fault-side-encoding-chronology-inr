import pandas as pd
import numpy as np
import pyvista as pv
import torch

def normalize(data, bounds):
    """
    Normalize the data to [-1, 1]
    data: the data to be normalized
    bounds: the extent of the model domain
    """
    data = data.astype(np.float32)
    mean_x = (bounds[0] + bounds[1]) / 2
    mean_y = (bounds[2] + bounds[3]) / 2
    mean_z = (bounds[4] + bounds[5]) / 2
    delta_x = bounds[1] - bounds[0]
    delta_y = bounds[3] - bounds[2]
    delta_z = bounds[5] - bounds[4]
    data[:, 0] = (data[:, 0] - mean_x) / delta_x * 2
    data[:, 1] = (data[:, 1] - mean_y) / delta_y * 2
    data[:, 2] = (data[:, 2] - mean_z) / delta_z * 2

    return data


def extend_surface(surface_point):
    """
    Notes: 1. extend the surface points to the above and below to generate three surfaces
           2. after the normalization, the surface points are in the range of [-1, 1], set the extend distance as 0.2
           3. the original surface points are in the middle, scalar value is 0, the above surface is 1, the below surface is -1
    surface_point: the surface points
    return: the extended surface points coordinates and the corresponding labels
    """
    surface_point0 = surface_point.copy()
    surface_point2 = surface_point.copy()
    surface_point0[:,2] = surface_point0[:,2] - 0.2
    surface_point2[:,2] = surface_point2[:,2] + 0.2
    y_tensor_0 = np.ones(len(surface_point0[:,0]))*(-1)
    y_tensor_1 = np.zeros(len(surface_point[:,0]))
    y_tensor_2 = np.ones(len(surface_point2[:,0]))
    points = np.vstack([surface_point0, surface_point, surface_point2])
    labels = np.hstack([y_tensor_0, y_tensor_1, y_tensor_2])

    return points, labels


def query_points(bounds, resolution):
    """
    Generate meshgrid points for the model, which is used to inference the domain
    *** only use for fault surface generation, the boundary needs to extend a cell_size to avoid error in clip_surface (feature encoding step)
    bounds: the extent of the model domain
    resolution: the resolution of the model
    """
    cell_size = (bounds[1] - bounds[0]) / resolution[0]
    x = np.linspace(bounds[0]-cell_size, bounds[1]+cell_size, resolution[0])
    y = np.linspace(bounds[2]-cell_size, bounds[3]+cell_size, resolution[1])
    z = np.linspace(bounds[4]-cell_size, bounds[5]+cell_size, resolution[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    test_x = np.vstack((xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F'))).T

    return test_x

def domain_meshgrid(extents, resolution):
        """
        a defined grid mesh for the model domain
        extents: the extent of the model domain
        resolution: the resolution of the model
        """
        grid_mesh_final = pv.ImageData()
        grid_mesh_final.dimensions = resolution
        grid_mesh_final.origin = [extents[0], extents[2], extents[4]]
        grid_mesh_final.spacing = [(extents[1]-extents[0])/(resolution[0]-1), (extents[3]-extents[2])/(resolution[1]-1), 
                                (extents[5]-extents[4])/(resolution[2]-1)]

        return grid_mesh_final


def query_points_exect_bounds(bounds, resolution):
    """
    Generate meshgrid points for the model, which is used to inference the domain
    *** use for except fault surface generation, the boundary is the same as the model domain
    bounds: the extent of the model domain
    resolution: the resolution of the model
    """
    x = np.linspace(bounds[0], bounds[1], resolution[0])
    y = np.linspace(bounds[2], bounds[3], resolution[1])
    z = np.linspace(bounds[4], bounds[5], resolution[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    test_x = np.vstack((xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F'))).T

    return test_x


def domain_meshgrid(extents, resolution):
    """
    define the meshgrid points for the model domain, this function used in 'feature_encoding.clip_surface'
    extents: the extent of the model domain
    resolution: the resolution of the model
    """
    grid_mesh_final = pv.ImageData()
    grid_mesh_final.dimensions = resolution
    grid_mesh_final.origin = [extents[0], extents[2], extents[4]]
    grid_mesh_final.spacing = [(extents[1]-extents[0])/(resolution[0]-1), (extents[3]-extents[2])/(resolution[1]-1), 
                            (extents[5]-extents[4])/(resolution[2]-1)]

    return grid_mesh_final


def predict_to_mesh_fault(extents, resolution, predictions):
    """
    the prediction results of gridmesh points are assigned back to the gridmesh points under the same resolution
    then extract the contour of the gridmesh points (isosurface=0) to generate the fault surface
    extents: the extent of the model domain
    resolution: the resolution of the model
    predictions: the prediction results of the gridmesh points
    """
    extents = extents.copy()
    cell_size = (extents[1] - extents[0]) / resolution[0]
    extents[0] = extents[0] - cell_size
    extents[1] = extents[1] + cell_size
    extents[2] = extents[2] - cell_size
    extents[3] = extents[3] + cell_size
    extents[4] = extents[4] - cell_size
    extents[5] = extents[5] + cell_size
    grid_mesh_final = pv.ImageData()
    grid_mesh_final.dimensions = resolution
    grid_mesh_final.origin = [extents[0], extents[2], extents[4]]
    grid_mesh_final.spacing = [(extents[1]-extents[0])/(resolution[0]-1), (extents[3]-extents[2])/(resolution[1]-1), (extents[5]-extents[4])/(resolution[2]-1)]

    grid_mesh_final.point_data['scalar'] = predictions.ravel()
    contours_mean = grid_mesh_final.contour(isosurfaces=[0]) 

    return contours_mean

def predict_to_mesh_single_surface(extents, resolution, predictions):
    """
    the prediction results of gridmesh points are assigned back to the gridmesh points under the same resolution
    then extract the contour of the gridmesh points (isosurface=0) to generate the fault surface
    extents: the extent of the model domain
    resolution: the resolution of the model
    predictions: the prediction results of the gridmesh points
    """
    grid_mesh_final = pv.ImageData()
    grid_mesh_final.dimensions = resolution
    grid_mesh_final.origin = [extents[0], extents[2], extents[4]]
    grid_mesh_final.spacing = [(extents[1]-extents[0])/(resolution[0]-1), (extents[3]-extents[2])/(resolution[1]-1), (extents[5]-extents[4])/(resolution[2]-1)]

    grid_mesh_final.point_data['scalar'] = predictions.ravel()
    contours_mean = grid_mesh_final.contour(isosurfaces=[0]) 

    return contours_mean, grid_mesh_final


def predict_to_mesh_stratigraphic(extents, resolution, predictions, iso_values):
    """
    the prediction results of gridmesh points are assigned back to the gridmesh points under the same resolution
    multiple isovalues are used to generate the stratigraphic surfaces according to the 'iso_values'
    extents: the extent of the model domain
    resolution: the resolution of the model
    predictions: the prediction results of the gridmesh points
    """
    grid_mesh_final = pv.ImageData()
    grid_mesh_final.dimensions = resolution
    grid_mesh_final.origin = [extents[0], extents[2], extents[4]]
    grid_mesh_final.spacing = [(extents[1]-extents[0])/(resolution[0]-1), (extents[3]-extents[2])/(resolution[1]-1), (extents[5]-extents[4])/(resolution[2]-1)]

    grid_mesh_final.point_data['scalar'] = predictions.ravel()
    contours_mean = grid_mesh_final.contour(isosurfaces=iso_values) 

    return contours_mean, grid_mesh_final

def build_volume_model_from_scalar(scalar_field, output_path=None, thresholds=None, label_name='Rock types'):
    if thresholds is None:
        thresholds = (-1.0, -0.33, 0.33, 1.0)
    t0, t1, t2, t3 = thresholds

    volume_mesh = scalar_field.copy(deep=True)
    scalar = volume_mesh.point_data['scalar']
    labels = np.zeros_like(scalar, dtype=int)
    labels = np.where(scalar >= t0, 1, labels)
    labels = np.where(scalar >= t1, 2, labels)
    labels = np.where(scalar >= t2, 3, labels)
    labels = np.where(scalar >= t3, 4, labels)
    volume_mesh.point_data[label_name] = labels.ravel()

    if output_path:
        volume_mesh.save(output_path)

    return volume_mesh


# generate the additional points for computing orientation
def delta_cal_orie(vector_points, resolution, delta):
    """
    Note: 1. generate the additional points for computing orientation, oneorientation point will generate 6 additional points,
          2. the additional 6 points have the same fault feature with the original orientation point
    vector_points: the input points for computing orientation
    resolution: the resolution of the model
    delta: the distance for computing orientation, usually set 1 cell size
    """
    vector_points = vector_points.copy()
    extents = [-1, 1, -1, 1, -1, 1]
    deltax = (extents[1] - extents[0]) / resolution[0]
    deltay = (extents[3] - extents[2]) / resolution[1]
    deltaz = (extents[5] - extents[4]) / resolution[2]
    tensor_x_plus = vector_points.copy()
    tensor_x_plus[:,0] = tensor_x_plus[:,0] + deltax*delta
    tensor_x_minus = vector_points.copy()
    tensor_x_minus[:,0] = tensor_x_minus[:,0] - deltax*delta
    tensor_y_plus = vector_points.copy()
    tensor_y_plus[:,1] = tensor_y_plus[:,1] + deltay*delta
    tensor_y_minus = vector_points.copy()
    tensor_y_minus[:,1] = tensor_y_minus[:,1] - deltay*delta
    tensor_z_plus = vector_points.copy()
    tensor_z_plus[:,2] = tensor_z_plus[:,2] + deltaz*delta
    tensor_z_minus = vector_points.copy()
    tensor_z_minus[:,2] = tensor_z_minus[:,2] - deltaz*delta
    new_vector_points = np.vstack((tensor_x_plus, tensor_x_minus, tensor_y_plus, tensor_y_minus, tensor_z_plus, tensor_z_minus))
    return new_vector_points


# define the gradient loss function
def calculate_grad_loss(grad_pred, normals, method='sum'):
    """
    calculate the gradient loss between the predicted gradient and the true normal vectors, cosine similarity is used
    grad_pred: the predicted gradient
    normals: the true normal vectors
    method: the method to calculate the loss, 'sum' or 'mean'
    """
    grad_norm_pred = torch.norm(grad_pred, p=2, dim=1)
    grad_inner_product = torch.einsum('ij, ij->i', normals, grad_pred)
    cosine = grad_inner_product / grad_norm_pred # cosine similarity
    if method == 'sum':
        return torch.sum(1 - cosine)
    else:  # sum
        return torch.mean(1 - cosine)


def compute_gradient(input, n, resolution, delta):
    """
    compute the gradient of the input data
    input: the predicted scalar value of additional orientation points (6*n_orie)
    n: the number of orientation points
    resolution: the resolution of the model
    delta: the distance for computing the gradient
    """
    extents = [-1, 1, -1, 1, -1, 1]  
    deltax = (extents[1] - extents[0]) / resolution[0]
    deltay = (extents[3] - extents[2]) / resolution[1]
    deltaz = (extents[5] - extents[4]) / resolution[2]
    df_dx = (input[0:n] - input[n:2*n]) / (2 * deltax * delta)
    df_dy = (input[2*n:3*n] - input[3*n:4*n]) / (2 * deltay * delta)
    df_dz = (input[4*n:5*n] - input[5*n:6*n]) / (2 * deltaz * delta)
    return torch.hstack((df_dx, df_dy, df_dz))
