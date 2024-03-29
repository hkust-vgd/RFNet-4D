import torch
import numpy as np
import trimesh
import os
from im2mesh.common import load_and_scale_mesh, load_and_scale_mesh_sequence, load_and_scale_pointcloud_sequence
from im2mesh.utils.io import save_mesh
import time
from im2mesh.utils.onet_generator import Generator3D as Generator3DONet
import pandas as pd


class Generator3D(Generator3DONet):
    ''' UnFlow Generator object class.

    It provides methods to extract final meshes from the UnFlow representation.

    Args:
        model (nn.Module): UnFlow model
        device (device): PyTorch device
        points_batch_size (int): batch size for evaluation points to extract
            the shape at t=0
        threshold (float): threshold value for the Occupancy Networks-based
            shape representation at t=0
        refinement_step (int): number of refinement step for MISE
        padding (float): padding value for MISE
        sample (bool): whether to sample from prior for latent code z
        simplify_nfaces (int): number of faces the mesh should be simplified to
        n_time_steps (int): number of time steps which should be extracted
        mesh_color (bool): whether to save the meshes with color
            encoding
        only_end_time_points (bool): whether to only generate first and last
            mesh
        interpolate (bool): whether to use the velocity field to interpolate
            between start and end mesh
        fix_z (bool): whether to hold latent shape code fixed
        fix_zt (bool): whether to hold latent motion code fixed
    '''

    def __init__(self, model, device=None, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, resolution0=16,
                 upsampling_steps=3, padding=0.1, sample=False,
                 simplify_nfaces=None, n_time_steps=17,  
                 mesh_color=False, only_end_time_points=False, interpolate=False,
                 fix_z=False, fix_zt=False, **kwargs):

        self.n_time_steps = n_time_steps
        self.mesh_color = mesh_color
        self.only_end_time_points = only_end_time_points
        self.interpolate = interpolate
        self.fix_z = fix_z
        self.fix_zt = fix_zt

        self.onet_generator = Generator3DONet(
            model, device=device, points_batch_size=points_batch_size,
            threshold=threshold, refinement_step=refinement_step,
            resolution0=resolution0, upsampling_steps=upsampling_steps,
            with_normals=False, padding=padding, sample=sample,
            simplify_nfaces=simplify_nfaces)

        if fix_z:
            self.fixed_z, _ = self.onet_generator.model.get_z_from_prior(
                (1,), sample=sample)
        if fix_zt:
            _, self.fixed_zt = self.onet_generator.model.get_z_from_prior(
                (1,), sample=sample)

    def return_face_colors(self, n_faces):
        ''' Returns the face colors.

        Args:
            n_faces (int): number of faces
        '''
        if self.mesh_color:
            step_size = 255. / n_faces
            colors = [[int(255 - i*step_size), 25, int(i*step_size), 255]
                      for i in range(n_faces)]
            colors = np.array(colors).astype(np.uint64)
        else:
            colors = None
        return colors

    def generate_mesh_t0(self, z=None, c_s=None, c_t=None, data=None,
                         stats_dict={}):
        ''' Generates the mesh at time step t=0.

        Args:
            z (tensor): latent code z
            c_s (tensor): conditioned spatial code c_s
            c_t (tensor): conditioned temporal code c_t
            data (dict): data dictionary
            stats_dict (dict): (time) statistics dictionary
        '''
        if self.onet_generator.model.decoder is not None:
            if len(c_s.size()) == 3:
                c_s = c_s[:, 0, :]
            mesh = self.onet_generator.generate_from_latent(
                z, c_s, c_t, stats_dict=stats_dict)
        else:
            vertices = data['mesh.vertices'][:, 0].squeeze(0).cpu().numpy()
            faces = data['mesh.faces'].squeeze(0).cpu().numpy()
            mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, process=False)
        return mesh

    def get_time_steps(self):
        ''' Returns time steps for mesh extraction.
        '''
        n_steps = self.n_time_steps
        device = self.onet_generator.device

        if self.only_end_time_points:
            t = torch.tensor([0., 1.]).to(device)
        else:
            t = (torch.arange(1, n_steps).float() / (n_steps - 1)).to(device)

        return t

    def generate_meshes_t(self, vertices_0, faces, z=None, c_s=None, c_t=None,
                          vertex_data=None, stats_dict={}):
        ''' Generates meshes for time steps t>0.

        Args:
            vertices_0 (numpy array): vertices of mesh at t=0
            faces (numpy array): faces of mesh at t=0
            z (tensor): latent code z
            c_t (tensor): temporal conditioned code c_t
            vertex_data (tensor): vertex tensor (start and end mesh if
                interpolation is required)
            stats_dict (dict): (time) statistics dictionary
        '''
        device = self.onet_generator.device
        time_val = self.get_time_steps()
        meshes = []
        vertices_0 = torch.from_numpy(
            vertices_0).to(device).unsqueeze(0).float()
        t0 = time.time()

        #Parallel Iso-Surface Deformation
        n_steps = len(time_val)
        c_s=c_s[:,1:,:]

        v_t_batch, flow_features = self.onet_generator.model.transform_to_t(time_val, vertices_0.expand(n_steps, vertices_0.size(1), vertices_0.size(2)), \
            c_t=c_t.expand(n_steps, c_t.size(1), c_t.size(2)))
        v_t_batch = v_t_batch.cpu().numpy()

        stats_dict['time (forward propagation)'] = time.time() - t0

        for v_t in v_t_batch:
            mesh = trimesh.Trimesh(vertices=v_t, faces=faces, process=False)
            meshes.append(mesh)
        return meshes

    def export(self, meshes, mesh_dir, modelname, start_idx=0, start_id_seq=1):
        ''' Exports a list of meshes.

        Args:
            meshes (list): list of trimesh meshes
            mesh_dir (str): mesh directory
            modelname (str): model name
            start_idx (int): start id of sequence (for naming convention)
            start_id_seq (int): id of start mesh in its sequence
                (e.g. 1 for start mesh)
        '''
        model_folder = os.path.join(mesh_dir, modelname, '%05d' % start_idx)
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)

        return self.export_multiple_meshes(
            meshes, model_folder, modelname, start_idx,
            start_id_seq=start_id_seq)

    def export_mesh(self, mesh, model_folder, modelname, start_idx=0, n_id=1):
        ''' Exports a mesh.

        Args:
            mesh (trimesh): trimesh mesh object
            model_folder (str): model folder
            modelname (str): model name
            n_id (int): number of time step (for naming convention)
        '''
        colors = self.return_face_colors(len(mesh.faces))
        out_path = os.path.join(
            model_folder, '%s_%04d_%04d.off' % (modelname, start_idx, n_id))
        save_mesh(mesh, out_path, face_colors=colors)
        return out_path

    def export_multiple_meshes(self, meshes, model_folder, modelname,
                               start_idx=0, start_id_seq=2):
        ''' Exports multiple meshes for consecutive time steps.

        Args:
            meshes (list): list of meshes
            model_folder (str): model folder
            modelname (str): model name
            start_id_seq (int): id of start mesh in its sequence
                (e.g. 1 for start mesh)        '''
        out_files = []
        for i, m in enumerate(meshes):
            out_files.append(self.export_mesh(
                m, model_folder, modelname, start_idx, n_id=start_id_seq + i))
        return out_files


    def generate(self, data, return_stats=True, n_samples=1, **kwargs):
        ''' Generates meshes for respective time steps.

        Args:
            data (dict): data dictionary
            mesh_dir (str): mesh directory
            modelname (str): model name
            return_stats (bool): whether to return (time) statistics
            n_samples (int): number of latent samples which should be used
        '''
        self.onet_generator.model.eval()
        stats_dict = {}
        device = self.onet_generator.device
        inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)
        vertex_data = data.get('mesh.vertices', None)

        meshes = []
        with torch.no_grad():
            for i in range(n_samples):
                t0 = time.time()
                c_s, c_t = self.onet_generator.model.encode_inputs(inputs)
                stats_dict['time (encode inputs)'] = time.time() - t0
                
                z, z_t = self.onet_generator.model.get_z_from_prior(
                    (1,), sample=self.onet_generator.sample)
                if self.fix_z:
                    z = self.fixed_z
                if self.fix_zt:
                    z_t = self.fixed_zt
                z = z.to(device)
                z_t = z_t.to(device)

                # Generate and save first mesh
                mesh_t0 = self.generate_mesh_t0(z, c_s, c_t, data, stats_dict=stats_dict)
                meshes.append(mesh_t0)

                # Generate and save later time steps
                meshes_t = self.generate_meshes_t(
                    mesh_t0.vertices, mesh_t0.faces, z=z, c_s=c_s, c_t=c_t,
                    vertex_data=vertex_data, stats_dict=stats_dict)
                meshes.extend(meshes_t)
        return meshes, stats_dict


    ##### firstly shape interpolation, and then add motion
    def generate_latent_space_interpolation(self, model_0, model_1,
                                            latent_space_file_path=None,
                                            n_samples=2, **kwargs):
        ''' Generates a latent space interpolation.

        For usage, check generate_latent_space_interpolation.py.

        Args:
            model_0 (dict): dictionary for model_0
            model_1 (dict): dictionary for model_1
            latent_space_file_path (str): path to latent space file
            n_samples (int): how many samples to generate between start and end
        '''
        self.onet_generator.model.eval()
        device = self.onet_generator.device
        assert(self.fix_z != self.fix_zt)

        # Get latent codes for interpolation
        df = pd.read_pickle(latent_space_file_path)
        k_interpolate = 'loc_z_t' if self.fix_z else 'loc_z'
        z0 = torch.from_numpy(df.loc[(df['model'] == model_0['model']) & (
            df['start_idx'] == model_0['start_idx'])][k_interpolate].item()
        ).unsqueeze(0).to(device)
        zt = torch.from_numpy(df.loc[(df['model'] == model_1['model']) & (
            df['start_idx'] == model_1['start_idx'])][k_interpolate].item()
        ).unsqueeze(0).to(device)

        # Get fixed latent code from start mesh
        k_fixed = 'loc_z' if self.fix_z else 'loc_z'
        fixed_z = torch.from_numpy(
            df.loc[((df['model'] == model_0['model']) &
                    (df['start_idx'] == model_0['start_idx']))][k_fixed].item()
        ).unsqueeze(0).to(device)

        # Not use additional input
        c_s = torch.empty(1, 0).to(device)
        c_t = torch.empty(1, 0).to(device)
        stats_dict = {}
        meshes = []
        with torch.no_grad():
            for i in range(n_samples):
                t0 = time.time()
                zi = zt * (i/(n_samples-1)) + z0 * (1 - (i/(n_samples-1)))
                if self.fix_z:
                    z, z_t = fixed_z, zi
                else:
                    z_t, z = fixed_z, zi
                stats_dict['time (encode inputs)'] = time.time() - t0

                # Generate and save first mesh
                mesh_t0 = self.generate_mesh_t0(
                    z, c_s, c_t, None, stats_dict=stats_dict)
                meshes.append(mesh_t0)
                # Generate and save later time steps
                meshes_t = self.generate_meshes_t(
                    mesh_t0.vertices, mesh_t0.faces, z=z_t, c_t=c_t,
                    stats_dict=stats_dict)
                meshes.extend(meshes_t)
        return meshes, stats_dict
    #####

    ##### motion transfer
    def generate_motion_transfer(self, model_0, model_1, motion_model_path, motion_pointcloud_path, shape_file_path):
        self.onet_generator.model.eval()
        device = self.onet_generator.device
        vertex_data = None

        #Load Motion
        _, _, motion_mesh_vertices = load_and_scale_mesh_sequence(motion_model_path, model_0['start_idx'], seq_len=17)
        _, _, motion_pointclouds = load_and_scale_pointcloud_sequence(motion_pointcloud_path, model_0['start_idx'], seq_len=17)

        inputs = torch.from_numpy(motion_pointclouds[None, ...]).to(device)
        with torch.no_grad():
            c_s, c_t = self.onet_generator.model.encode_inputs(inputs)

        meshes = []
        # Load Shape 
        mesh_t0_path = os.path.join(
            shape_file_path, '%08d.obj' % model_1['start_idx']) #%05d 08d
        _, _, mesh_t0 = load_and_scale_mesh(mesh_t0_path)
        meshes.append(mesh_t0)

        # Generate and save later time steps
        with torch.no_grad():
            meshes_t = self.generate_meshes_t(
                mesh_t0.vertices, mesh_t0.faces, c_t=c_t,
                vertex_data=vertex_data)
            meshes.extend(meshes_t)
        return meshes

    ################
    def interpolate_sequence(self, data, n_time_steps=17, **kwargs):
        ''' Generates an interpolation sequence.

        Args:
            data (dict): data dictionary
            mesh_dir (str): mesh directory
            modelname (str): model name
            n_time_steps (int): number of time steps which should be generated.
                If no value is passed, the standard object value for
                n_time_steps is used.
        '''
        self.onet_generator.model.eval()
        device = self.onet_generator.device
        inputs_full = data.get('inputs', torch.empty(1, 1, 0)).to(device)
        faces = data['mesh.faces'].squeeze(0).cpu().numpy()
        vertices = data['mesh.vertices']
        num_files = inputs_full.shape[0]

        self.n_time_steps = n_time_steps

        meshes = []

        # Save first mesh
        mesh_0 = trimesh.Trimesh(
            vertices=vertices[0].cpu().numpy(), faces=faces, process=False)
        meshes.append(mesh_0)

        # Save later time steps
        for i in range(num_files - 1):
            inputs = inputs_full[i:i+2].unsqueeze(0)
            vertex_data = data['mesh.vertices'][i:i+2].unsqueeze(0)

            mesh_t0_vertices = vertices[i].cpu().numpy()

            with torch.no_grad():
                _, c_t = self.onet_generator.model.encode_inputs(inputs)
                c_t_list, c_t_first, c_t_last = [], c_t[:, 0, :], c_t[:, -1, :]
                for i in range(n_time_steps):
                    c_t_new = c_t_last * (i/(n_time_steps-1)) + c_t_first * (1 - (i/(n_time_steps-1)))
                    c_t_list.append(c_t_new)
                c_t = torch.stack(c_t_list, dim=1)
                meshes_t = self.generate_meshes_t(mesh_t0_vertices, faces, c_t=c_t,
                                                  vertex_data=vertex_data)
                meshes.extend(meshes_t)
        return meshes
