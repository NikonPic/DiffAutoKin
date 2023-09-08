# %%
from datetime import datetime
import os
from matplotlib import pyplot as plt
import torch
from torch import nn
import xml.etree.ElementTree as ET
from tqdm import tqdm
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import hydra
from torch.utils.tensorboard import SummaryWriter
from hydra.core.config_store import ConfigStore
from config import DiffJointTraining
from dm_control import mujoco


cs = ConfigStore.instance()
cs.store(name='training_config', node=DiffJointTraining)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Joint(nn.Module):
    def __init__(self, pos_init, axis_init, device, max_diff_norm=1.5):
        super(Joint, self).__init__()
        assert len(pos_init) == 3 and len(
            axis_init) == 3, "Inputs pos and axis should be 3-element list or tensor."
        self.max_diff_norm = max_diff_norm
        self.pos_init = nn.Parameter(torch.tensor(
            pos_init, dtype=torch.float32, device=device), requires_grad=False)
        self.axis_init = nn.Parameter(torch.tensor(
            axis_init, dtype=torch.float32, device=device), requires_grad=False)
        self.pos_diff = nn.Parameter(torch.zeros(
            3, dtype=torch.float32, device=device), requires_grad=True)
        self.axis_diff = nn.Parameter(torch.zeros(
            3, dtype=torch.float32, device=device), requires_grad=True)
        self.device = device

    @property
    def pos(self):
        diff_norm = torch.norm(self.pos_diff)
        if diff_norm > self.max_diff_norm:
            self.pos_diff.data = self.max_diff_norm * self.pos_diff.data / diff_norm
        return self.pos_init + self.pos_diff

    @property
    def axis(self):
        diff_norm = torch.norm(self.axis_diff)
        if diff_norm > self.max_diff_norm:
            self.axis_diff.data = self.max_diff_norm * self.axis_diff.data / diff_norm
        return (self.axis_init + self.axis_diff) / torch.norm(self.axis_init + self.axis_diff)

    def forward(self, theta, T_prev):
        """Forward propagation through the joint."""
        n_timesteps = T_prev.shape[0]

        # Make sure the axis is normalized
        axis_normalized = self.axis / torch.norm(self.axis)

        # Compute the rotation matrix
        c = torch.cos(theta).reshape(-1)
        s = torch.sin(theta).reshape(-1)
        C = 1 - c
        x = axis_normalized[0]
        y = axis_normalized[1]
        z = axis_normalized[2]
        xs = x * s
        ys = y * s
        zs = z * s
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        row1 = torch.stack([x*xC + c,   xyC - zs,   zxC + ys], dim=1)
        row2 = torch.stack([xyC + zs,   y*yC + c,   yzC - xs], dim=1)
        row3 = torch.stack([zxC - ys,   yzC + xs,   z*zC + c], dim=1)
        rotation = torch.stack([row1, row2, row3], dim=1)

        # Compute the transformation matrix for the rotation
        T_rot = torch.eye(4, device=T_prev.device).unsqueeze(
            0).repeat(n_timesteps, 1, 1)
        T_rot[:, :3, :3] = rotation

        # Compute the translation matrices
        T_trans = torch.eye(4, device=T_prev.device).unsqueeze(
            0).repeat(n_timesteps, 1, 1)
        T_trans_inv = T_trans.clone()
        T_trans[:, :3, 3] = self.pos.unsqueeze(0)
        T_trans_inv[:, :3, 3] = -self.pos.unsqueeze(0)

        # First translate to the origin, then rotate, then translate back
        T_new = T_prev @ T_trans @ T_rot @ T_trans_inv
        return T_new


class RigidBody(nn.Module):
    def __init__(self, name, parent_name, joint_name, pos, quat, device, joint=None, joint_idx=None, subsequent_bodies=None):
        super(RigidBody, self).__init__()
        self.name = name
        self.parent_name = parent_name  # The name of the parent body
        self.joint_name = joint_name
        self.pos = pos.to(device)
        self.quat = quat.to(device)
        self.device = device
        self.joint = joint
        self.joint_idx = joint_idx
        self.subsequent_bodies = subsequent_bodies if subsequent_bodies is not None else []
        self.rot_matrix = self.quat_to_rot_matrix()  # Pre-calculate the rotation matrix

    def __str__(self):
        return_str = f"RigidBody(name={self.name}, parent_name={self.parent_name})\n"
        if self.joint is not None:
            axis_str = f'{self.joint.axis}'.replace(',', '')
            pos_str = f'{self.joint.pos}'.replace(',', '')
            return_str = f'{return_str}  {self.joint_name}\n    Axis: {axis_str}\n'
            return_str = f'{return_str}    Pos: {pos_str}\n\n'

        return return_str

    def forward(self, theta, T_prev):
        id_mat = torch.eye(4).repeat(T_prev.shape[0], 1, 1).to(self.device)

        if self.joint is not None and theta is not None:
            # Joint transformation
            T_joint = self.joint(theta, id_mat)
        else:
            T_joint = id_mat

        # Body transformation
        T_body = torch.zeros_like(T_prev)
        # Use the pre-calculated rotation matrix
        T_body[:, :3, :3] = self.rot_matrix
        T_body[:, :3, 3] = self.pos
        T_body[:, 3, 3] = 1

        return T_prev @ T_body @ T_joint

    def quat_to_rot_matrix(self):
        """Convert quaternion to rotation matrix."""
        q0, q1, q2, q3 = torch.split(self.quat, 1)
        R = torch.zeros((3, 3), device=self.device)
        R[0, 0] = 1 - 2 * (q2**2 + q3**2)
        R[0, 1] = 2 * (q1*q2 - q0*q3)
        R[0, 2] = 2 * (q1*q3 + q0*q2)
        R[1, 0] = 2 * (q1*q2 + q0*q3)
        R[1, 1] = 1 - 2 * (q1**2 + q3**2)
        R[1, 2] = 2 * (q2*q3 - q0*q1)
        R[2, 0] = 2 * (q1*q3 - q0*q2)
        R[2, 1] = 2 * (q2*q3 + q0*q1)
        R[2, 2] = 1 - 2 * (q1**2 + q2**2)
        return R


class ChainKinematics(nn.Module):
    def __init__(self, bodies, device):
        super(ChainKinematics, self).__init__()
        self.bodies = nn.ModuleList(bodies)  # The bodies in the chain
        # The device (CPU or GPU) to perform computations on
        self.device = device

    def forward(self, theta, key_names):
        # Create an identity matrix for reuse
        id_mat = torch.eye(4).repeat(theta.shape[0], 1, 1).to(self.device)

        # Create dictionaries to store the transformation matrix of each body
        transformations = {"root": id_mat}
        transformations_rel = {"root": id_mat}

        # Propagate transformations through the chain
        for body in self.bodies:
            self.forward_body(body, theta, transformations,
                              transformations_rel, key_names)

        return transformations, transformations_rel

    def forward_body(self, body, theta, transformations, transformations_rel, key_names):
        """Recursive function to propagate the transformations through the kinematic tree."""
        joint_idx = body.joint_idx  # Each body should know which joint it's associated with

        # Get joint angle for the current body (if there's no joint associated, then theta is None)
        theta_curr = theta[:, joint_idx:joint_idx +
                           1] if joint_idx is not None else None

        # Get root and parent transformations
        T_root = transformations['root']
        T_parent = transformations[body.parent_name]

        # Calculate the current transformation matrix relative to the root
        T_curr = body(theta_curr, T_root)

        # Store the transformation matrix in transformations
        transformations[body.name] = T_parent @ T_curr

        # If the parent name is in key names, the relative transformation is the same as the current
        if body.parent_name in key_names:
            transformations_rel[body.name] = T_curr
        else:
            # Calculate and store the relative transformation
            T_prev = transformations_rel[body.parent_name]
            transformations_rel[body.name] = T_prev @ T_curr


class ChainInitializer:
    def __init__(self, xml_path, yaml_path):
        self.xml_path = xml_path
        self.yaml_path = yaml_path

    def get_parent_map(self, tree):
        return {c: p for p in tree.iter() for c in p}

    def parse_xml(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Create parent map
        parent_map = self.get_parent_map(tree)

        body_inits = []
        joint_idx = 0

        for body in root.iter('body'):
            pos_str = body.get('pos')
            quat_str = body.get('quat')
            body_name = body.get('name')

            pos = torch.tensor([float(x) for x in pos_str.split()])
            quat = torch.tensor([float(x) for x in quat_str.split()])

            joint = body.find('joint')
            joint_name = joint.get('name') if joint is not None else None

            # Get the parent body's name
            parent_body = parent_map.get(body)

            # If parent body is "worldbody", assign "root" as its name
            if parent_body.tag == 'worldbody':
                parent_name = "root"
            else:
                parent_name = parent_body.get('name')

            if joint_name is not None:
                body_inits.append(
                    (pos, quat, body_name, parent_name, joint_name, joint_idx))
                joint_idx += 1
            else:
                body_inits.append(
                    (pos, quat, body_name, parent_name, None, None))

        return body_inits, joint_idx

    def parse_yaml(self):
        with open(self.yaml_path, 'r') as file:
            params = yaml.safe_load(file)

        return params

    def initialize_chain(self, device):
        body_inits, self.num_joints = self.parse_xml()
        params = self.parse_yaml()

        bodies = []
        for i in range(len(body_inits)):
            pos, quat, body_name, parent_name, joint_name, joint_idx = body_inits[i]
            if joint_name is not None:
                joint_params = params[joint_name]
                axis_init = torch.tensor(
                    [float(x) for x in joint_params['axis'].split()])
                pos_init = torch.tensor([float(x)
                                        for x in joint_params['pos'].split()])
                joint = Joint(pos_init, axis_init, device).to(device)
                bodies.append(RigidBody(body_name, parent_name, joint_name, pos, quat,
                              device, joint, joint_idx).to(device))
            else:
                bodies.append(RigidBody(body_name, parent_name, None, pos,
                              quat, device, None, None).to(device))

        return ChainKinematics(bodies, device)


class InverseKinematics(nn.Module):
    def __init__(self, input_dim, n_joints, hidden_dims=[256, 256], activation=nn.LeakyReLU):
        super(InverseKinematics, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            self.layers.append(activation())
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.layers.append(activation())
        self.layers.append(nn.Linear(hidden_dims[-1], n_joints))

    def forward(self, batch):
        # Reshape the input batch to match the input dimension of the model
        batch = batch.view(batch.size(0), -1)
        for layer in self.layers:
            batch = layer(batch)
        theta = torch.tanh(batch) * np.pi
        return theta


class HomogeneousTransformation(nn.Module):
    def __init__(self, pos, quat, device):
        super(HomogeneousTransformation, self).__init__()
        self.pos = nn.Parameter(pos.clone().detach().requires_grad_(True))
        self.quat = nn.Parameter(quat.clone().detach().requires_grad_(True))
        self.device = device

    def forward(self, x):
        T = self.get_transformation_matrix()
        return x @ T

    def get_transformation_matrix(self):
        """Calculate the 4x4 homogeneous transformation matrix."""
        # Calculate rotation matrix from quaternion
        R = self.quat_to_rot_matrix()

        # Create homogeneous transformation matrix
        T = torch.eye(4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = self.pos
        return T

    def quat_to_rot_matrix(self):
        """Convert quaternion to rotation matrix."""
        q0, q1, q2, q3 = torch.split(self.quat, 1)
        R = torch.zeros((3, 3), device=self.device)
        R[0, 0] = 1 - 2 * (q2**2 + q3**2)
        R[0, 1] = 2 * (q1*q2 - q0*q3)
        R[0, 2] = 2 * (q1*q3 + q0*q2)
        R[1, 0] = 2 * (q1*q2 + q0*q3)
        R[1, 1] = 1 - 2 * (q1**2 + q3**2)
        R[1, 2] = 2 * (q2*q3 - q0*q1)
        R[2, 0] = 2 * (q1*q3 - q0*q2)
        R[2, 1] = 2 * (q2*q3 + q0*q1)
        R[2, 2] = 1 - 2 * (q1**2 + q2**2)
        return R


class PoseDataset(Dataset):
    def __init__(self, dataset_path):
        super(PoseDataset, self).__init__()
        data_dict = np.load(dataset_path, allow_pickle=True).item()
        self.data_dict = {key: torch.tensor(val)
                          for key, val in data_dict.items()}
        self.keys = list(self.data_dict.keys())
        self.n_keys = len(self.keys)
        self.input_shape = self.n_keys * 16

    def __len__(self):
        # Assumes all arrays in the dictionary have the same number of time steps
        return self.data_dict[self.keys[0]].shape[0]

    def __getitem__(self, idx):
        return {key: self.data_dict[key][idx] for key in self.keys}


class Collator:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        # Initialize a dictionary to hold our batch data
        batched_data = {k: [] for k in batch[0].keys()}

        # Iterate over the batch
        for item in batch:
            for k, v in item.items():
                batched_data[k].append(v)

        # Convert each batch to a tensor
        for k, v in batched_data.items():
            batched_data[k] = torch.stack(v).to(self.device)

        return batched_data


def reconstruct_dict_from_batch(batch_tensor, keys):
    batch_size = batch_tensor.shape[0]
    n_keys = len(keys)
    each_key_elements = batch_tensor.shape[1] // n_keys

    dict_output = {}
    for j in range(n_keys):
        start = j * each_key_elements
        end = start + each_key_elements
        dict_output[keys[j]] = batch_tensor[:,
                                            start:end].view(batch_size, 4, 4)

    return dict_output


def calculate_mse(data_dict, transformations):
    """Calculate the Mean Squared Error (MSE) between the data dictionary and the transformations dictionary."""
    mse_loss = nn.MSELoss()
    mse_values = []

    # Iterate over all keys and calculate the MSE for each key
    for key in data_dict.keys():
        # Decompose the transformation matrix into a vector of size (batch_size, 7)
        trans_vec = transformations[key]

        # Calculate the MSE and append it to the list
        mse_values.append(mse_loss(data_dict[key], trans_vec))

    # Calculate the mean MSE over all keys using torch.stack
    mean_mse = torch.mean(torch.stack(mse_values))

    return mean_mse


class TrainingSchedule:
    def __init__(self, cfg: DiffJointTraining):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.dataset_path = cfg.data_path
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size
        self.device = torch.device(
            f'cuda:{cfg.cuda_device}' if torch.cuda.is_available() and cfg.use_cuda else 'cpu')

        # Initialize data
        self.dataset = PoseDataset(self.dataset_path)
        self.collator = Collator(self.device)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=cfg.shuffle,
            collate_fn=self.collator, num_workers=cfg.num_workers)
        self.keys = self.dataset.keys
        self.n_keys = len(self.keys)

        self.transf = HomogeneousTransformation(
            torch.tensor([0.0, 0.0, 0.0]), torch.tensor(
                [1.0, 0.0, 0.0, 0.0]), self.device
        ).to(self.device)

        chain_init = ChainInitializer(cfg.xml_path, cfg.yaml_path)
        self.decoder = chain_init.initialize_chain(self.device).to(self.device)
        self.num_joints = chain_init.num_joints
        self.encoder = InverseKinematics(
            self.dataset.input_shape, chain_init.num_joints, cfg.enc_hidden).to(self.device)

        # Initialize the optimizer and loss function
        self.optimizer_dec = optim.Adam(
            self.decoder.parameters(), lr=cfg.lr_dec)

        self.optimizer_enc = optim.Adam(
            self.encoder.parameters(), lr=cfg.lr_enc)

        self.optimizer_transf = optim.Adam(
            self.transf.parameters(), lr=cfg.lr_trafo)

        self.criterion = nn.MSELoss()
        self.setup_logging()
        self.saver = ChainSaver(cfg.load_path_xml, cfg.save_path_xml)
        self.learn_tmat = cfg.learn_tmat
        self.learn_relative = cfg.learn_relative
        self.fake = cfg.create_fake

        if self.cfg.load_previous:
            self.load_model()

        self.test_model()

    def create_fake_dataset(self):
        chain_init = ChainInitializer(self.cfg.xml_path, self.cfg.fake_params)
        self.decoder_fake = chain_init.initialize_chain(
            self.device).to(self.device)

    def setup_logging(self):
        now = datetime.now()
        if not os.path.exists(self.cfg.log_dir):
            os.makedirs(self.cfg.log_dir)
        self.dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        logdir = f'{self.cfg.log_dir}/{self.dt_string}'
        os.mkdir(logdir)
        self.writer = SummaryWriter(log_dir=logdir)

    def apply_transformation_to_batch(self, batch):
        transformed_batch = {}
        for key, item in batch.items():
            transformed_batch[key] = self.transf(item)
        return transformed_batch

    def print_infos(self):
        for body in self.decoder.bodies:
            print(body)
        print(self.transf.get_transformation_matrix())

    def batch_to_nn_input(self, batch):
        """Reshape and concatenate all tensors in the batch into a single 2D tensor."""
        batch_size = batch[self.keys[0]
                           ].shape[0]  # Get the batch size from the first key

        # Reshape each tensor to be 2D (batch_size, -1), then concatenate them all along dim=1
        input_tensor = torch.cat([batch[key].reshape(batch_size, -1)
                                  for key in self.keys], dim=1)

        # Check that the shape of the input tensor is as expected
        assert input_tensor.shape == (batch_size, self.n_keys * 4 * 4)
        return input_tensor

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.transf.train()

        train_count = 0
        running_loss = 0.0

        for epoch in tqdm(range(self.epochs)):

            for _, batch in enumerate(self.dataloader):
                train_count += 1
                # Zero the parameter gradients
                self.optimizer_enc.zero_grad()
                self.optimizer_dec.zero_grad()
                self.optimizer_transf.zero_grad()

                batch_flat = self.batch_to_nn_input(batch)

                if self.learn_tmat:
                    batch = self.apply_transformation_to_batch(batch)

                nn_input = self.batch_to_nn_input(batch)

                # Forward pass
                theta = self.encoder(nn_input)
                # Decode the joint angles to get the transformation matrices
                transformations, transformations_rel = self.decoder(
                    theta, self.keys)

                if self.learn_relative:
                    transformations = transformations_rel

                if self.learn_tmat:
                    transformations = self.apply_transformation_to_batch(
                        transformations)

                nn_output = self.batch_to_nn_input(transformations)

                # Compute loss
                loss = self.criterion(batch_flat, nn_output)
                running_loss += float(loss)

                # Backward pass and optimization
                loss.backward()
                self.optimizer_enc.step()
                self.optimizer_transf.step()

                if epoch > self.cfg.warmup_epochs:
                    self.optimizer_dec.step()

                # Print statistics
                running_loss += loss.item()
                if train_count == self.cfg.logging_intervall:  # print every x mini-batches
                    self.writer.add_scalar("main/loss", float(loss), epoch)
                    self.writer.add_scalar(
                        "main/running_loss", float(running_loss), epoch)
                    train_count = 0
                    running_loss = 0
                    self.create_joint_trajectories(f'{train_count}')

        self.print_infos()
        print('Finished Training')
        self.save_model()
        self.create_joint_trajectories('learned')

    def save_model(self):
        """Save the current state of the model to a file."""
        torch.save({
            'decoder': self.decoder.state_dict(),
            'encoder': self.encoder.state_dict(),
            'transf_state_dict': self.transf.state_dict(),
        }, self.cfg.save_path)
        self.saver.save(self.decoder)

    def load_model(self):
        """Load the model state from a file."""
        checkpoint = torch.load(self.cfg.load_path)
        self.decoder.load_state_dict(
            checkpoint['decoder'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.transf.load_state_dict(checkpoint['transf_state_dict'])

    def create_joint_trajectories(self, name='init'):
        """create trajectories of the joints using the learned inverse kinematics"""
        # use the whole dataset at once:
        loc_dataloader = DataLoader(self.dataset, batch_size=len(
            self.dataset), shuffle=False, collate_fn=self.collator)
        data = next(iter(loc_dataloader))
        dataset_flat = self.batch_to_nn_input(data)
        dataset_flat = dataset_flat.to(self.device)
        theta_whole = self.encoder(dataset_flat)
        theta_plot = theta_whole.detach().cpu().numpy() * (180 / np.pi)
        self.plot_joints(theta_plot, name)

    def plot_joints(self, theta_plot, name):

        if theta_plot.shape[1] > 3:
            labels = ['DAU CMC', 'DAU MCP', 'DAU PIP',
                    'DAU DIP', 'ZF MCP', 'ZF PIP', 'ZF DIP']
            plt.figure()
            ax1 = plt.subplot(211)
            for ind in range(0, 4):
                arr = theta_plot[:, ind] - np.min(theta_plot[:, ind])
                plt.plot(arr, label=labels[ind])
            plt.ylabel('angle in °')
            plt.legend()
            plt.grid(0.25)
            ax2 = plt.subplot(212, sharex=ax1)
            for ind in range(4, 7):
                arr = theta_plot[:, ind] - np.min(theta_plot[:, ind])
                plt.plot(arr, label=labels[ind])
            plt.legend()
            plt.grid(0.25)
            plt.xlabel('timesteps')
            plt.ylabel('angle in °')
            plt.tight_layout()
            plt.savefig(f'./results/thetas_{name}.png')
            plt.close()
        
        else:
            plt.figure()

            for ind in range(0, self.num_joints):
                arr = theta_plot[:, ind] - np.min(theta_plot[:, ind])
                plt.plot(arr)

            plt.grid(0.25)
            plt.xlabel('timesteps')
            plt.ylabel('angle in °')
            plt.savefig(f'./results/thetas_{name}.png')
            plt.close()


    def test_model(self):
        theta = np.random.randn(1, self.num_joints)
        self.physics = mujoco.Physics.from_xml_path(self.cfg.load_path_xml)
        self.physics.data.qpos = theta[0, :]
        self.physics.step()
        res_pt = self.decoder(torch.tensor(
            theta).to(self.device), self.keys)[0]

        for key in res_pt.keys():
            if key not in ['root', 'DAU_CMC', 'femur_0', 'femur_1']:
                pos_pt = res_pt[key][0, :3, 3].detach().cpu().numpy()
                pos_mj = torch.tensor(self.physics.data.body(key).xpos).numpy()
                print(f'{key}: {pos_pt} - {pos_mj}')


class ChainSaver:
    def __init__(self, xml_path, save_path):
        self.xml_path = xml_path
        self.save_path = save_path

    def update_xml(self, chain):
        # Parse the existing XML file
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Update the XML structure with learned parameters
        for body in root.iter('body'):
            # Find all joints of this body
            joints = body.findall('joint')

            # Find the matching body in the chain
            for rigid_body in chain.bodies:
                if body.get('name') == rigid_body.name:
                    # Update the position and orientation of the body
                    body.set('pos', ' '.join(
                        map(str, rigid_body.pos.detach().cpu().numpy())))
                    body.set('quat', ' '.join(
                        map(str, rigid_body.quat.detach().cpu().numpy())))

                # Iterate over all joints and update them
                for joint in joints:
                    if joint.get('name') == rigid_body.joint_name:
                        # Update the position and axis of the joint
                        joint.set('pos', ' '.join(
                                map(str, rigid_body.joint.pos.detach().cpu().numpy())))
                        joint.set('axis', ' '.join(
                                map(str, rigid_body.joint.axis.detach().cpu().numpy())))

        # Save the updated XML structure back to the file
        tree.write(self.save_path)

    def save(self, chain):
        self.update_xml(chain)


@hydra.main(version_base=None, config_path="../config", config_name="config_diff_joints_knee")
def main(cfg: DiffJointTraining):

    trainer = TrainingSchedule(cfg)
    trainer.print_infos()
    trainer.train()


# %%
if __name__ == '__main__':
    main()
    # %%
