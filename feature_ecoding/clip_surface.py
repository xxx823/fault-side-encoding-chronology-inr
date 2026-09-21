import pyvista as pv
import numpy as np
import pandas as pd


class Fault_feature():
    """
    Feature encoding for fault model
    """

    def domain_meshgrid(extents, resolution):
        """
        a defined grid mesh for the model domain
        extents: the extent of the model domain
        resolution: the resolution of the model
        """
        grid_mesh_final = pv.ImageData()  # ImageData replaces the deprecated UniformGrid.
        grid_mesh_final.dimensions = resolution
        grid_mesh_final.origin = [extents[0], extents[2], extents[4]]
        grid_mesh_final.spacing = [
            (extents[1] - extents[0]) / (resolution[0] - 1),
            (extents[3] - extents[2]) / (resolution[1] - 1),
            (extents[5] - extents[4]) / (resolution[2] - 1)
        ]
        return grid_mesh_final
    
    
    def feature_encoding_clip(fault_mesh, point_set, move, decimate, fault_direct='right'):
        """
        the main function for feature encoding, use computer implicit distance to assign labels to the points, method comes from the pyvista
        fault_mesh: the fault mesh to detemin which side the points belong to
        point_set: the points to be labeled
        move: the movement of the fault, 'up' or 'down'
        decimate: the decimate value for the fault mesh, to reduce the number of points
        fault_direct: the direction of the fault surface, 'right' or 'left'
        """
        decimated_mesh = fault_mesh.decimate_pro(decimate)  # if decimate=0.95, means only 5% of the points of the surface are kept
        point_set.compute_implicit_distance(decimated_mesh, inplace=True)
        implicit_distance = point_set.point_data['implicit_distance']
        # when fault_direct == 'right' and move == 'up', the points on right side are encoded as 1, the points on left side are encoded as 0
        # if fault_direct == 'right':
        #     flag = 1 if move == 'up' else 0
        #     encoding_value = np.where(implicit_distance < 0, flag, 1-flag)
        # else:
        #     flag = 0 if move == 'up' else 1
        #     encoding_value = np.where(implicit_distance < 0, flag, 1-flag)
        #
        # return encoding_value

        if fault_direct == 'right':
            flag = 1 if move == 'up' else 0
            encoding_value = np.where(implicit_distance < 0, flag, 1 - flag)

        elif fault_direct == 'left':
            flag = 1 if move == 'up' else 0
            # For the left convention, swap the binary assignments of the two sides.
            encoding_value = np.where(implicit_distance < 0, 1 - flag, flag)

        else:
            raise ValueError(f"fault_direct must be 'right' or 'left', got {fault_direct}")

        return encoding_value

    def encoding(extents, resolution, mesh_list, surface_points, orientation_points, movement=[], decimate=0.0, fault_direct=[]):
        """
        Note: 1. the main function for class Fault_feature, to encode the features for the fault model
              2. rule: when fault_direct == 'right' and move == 'up', the points on right side are encoded as 1, otherside is 0
              3. *** in the .csv file, the stratigraphic in 'formation' column should be in the order of old to new
        extents: the extent of the model domain
        resolution: the resolution of the model
        mesh_list: the list of the fault mesh
        surface_points: the surface points, from original data (.csv file)
        orientation_points: the orientation points, from original data (.csv file)
        movement: the movement relate to the fault, 'up' or 'down'
        decimate: the decimate value for the fault mesh, to reduce the number of points
        fault_direct: the direction of the fault surface, 'right' or 'left'
        """
        m = 0
        n = 0
        label_interf_all = []  # record the final fault feature encoding for the interface points, plus the label in the first column
        orie_points_all = []   # record the final fault feature encoding for the orientation points

        # domain mesh points
        domain_mesh = Fault_feature.domain_meshgrid(extents, resolution)
        domain_mesh_features = domain_mesh.points

        # interface points and orientation points
        # the labels in range [-1, 1], values related to the number of layers
        layer_name = surface_points[surface_points['type'] == 'stratigraphic']['formation'].unique()
        n_layer = len(layer_name)
        labels = np.linspace(-1, 1, n_layer)
        for layer in layer_name:
            coord = surface_points[surface_points['formation'] == layer][['X','Y','Z']].values
            coord_dx = orientation_points[orientation_points['formation'] == layer][['X','Y','Z','dx','dy','dz']].values
            # Formations must be ordered from older to younger before assigning labels.
            label_coord = np.insert(coord, 0, labels[m], axis=1)
            label_interf_all.append(label_coord)
            orie_points_all.append(coord_dx)
            m += 1
        label_interf_all = np.vstack(label_interf_all)
        orie_points_all = np.vstack(orie_points_all)
        # Polydata to mesh, function 'feature_encoding_clip' from pyvista requires the input as PolyData
        intf_points = pv.PolyData(label_interf_all[:,1:4])
        orie_points = pv.PolyData(orie_points_all[:,0:3])

        # Feature encoding
        for mesh in mesh_list:
            mesh = mesh.extract_surface()
            encoding_mesh_point = Fault_feature.feature_encoding_clip(mesh, domain_mesh, move=movement[n], decimate=decimate, fault_direct=fault_direct[n])
            domain_mesh_features = np.column_stack((domain_mesh_features, encoding_mesh_point))
            encoding_intf_point = Fault_feature.feature_encoding_clip(mesh, intf_points, move=movement[n], decimate=decimate, fault_direct=fault_direct[n])
            label_interf_all = np.column_stack((label_interf_all, encoding_intf_point))
            encoding_orie_point = Fault_feature.feature_encoding_clip(mesh, orie_points, move=movement[n], decimate=decimate, fault_direct=fault_direct[n])
            orie_points_all = np.column_stack((orie_points_all, encoding_orie_point))
            n += 1

        return domain_mesh_features, label_interf_all, orie_points_all
    
