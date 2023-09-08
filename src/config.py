# %% file to map the configs in yaml strucutre
from dataclasses import dataclass
from typing import List


@dataclass
class NetworkSettings:
    lr: float
    hidden_dims: List[int]
    n_input: int
    n_output: int
    with_std: bool


@dataclass
class ForKinSettings:
    lr: float
    n_output: int
    model_path: str
    scene: str
    parameter: str
    trained_parameters: str


@dataclass
class TrainingSettings:
    use_cuda: bool
    cuda_device: int
    seed: int
    epochs: int
    warmup_epochs: int
    logging_intervall: int
    data_path: str
    load_previous: bool
    load_path: str
    save_path: str
    log_dir: str
    batch_size: int
    num_workers: int
    shuffle: bool
    dec_for_model: ForKinSettings
    enc_inv_nn: NetworkSettings
    dec_for_nn: NetworkSettings


@dataclass
class DiffJointTraining:
    data_path: str
    xml_path: str
    yaml_path: str
    use_cuda: bool
    cuda_device: int
    seed: int
    epochs: int
    warmup_epochs: int
    logging_intervall: int
    load_previous: bool
    load_path: str
    save_path: str
    log_dir: str
    batch_size: int
    num_workers: int
    shuffle: bool
    lr_trafo: float
    lr_dec: float
    lr_enc: float
    enc_hidden: List[int]
    load_path_xml: str
    save_path_xml: str
    learn_tmat: bool
    learn_relative: bool
    create_fake: bool
    fake_params: str


@dataclass
class GenSettings:
    off_xyz: list
    off_wxyz: list
    marker_filt_threshold: float
    scale: float
    videoname: str
    number_opt_iters: int
    use_thumb_marker: bool
    fps: int
    start_pos: int
    end_pos: int


@dataclass
class TrackSettings:
    mj_name: str
    def_path: str
    ct_path: str
    stl_path: str
    csv_tracker_name: str
    idx: int
    list_num: int
    resort: bool
    resort_list: List[int]


@dataclass
class MarkerSettings:
    mj_name: str
    name: str
    ct_path_axis_distal: str
    ct_path_axis_proximal: str
    ct_path_marker: str
    stl_path: str
    start_id: str


@dataclass
class FingerAssignment:
    zf_strecker_1: int
    zf_strecker_2: int
    zf_beuger_1: int
    zf_beuger_2: int
    daumen_strecker_1: int
    daumen_strecker_2: int
    daumen_spreitzer: int
    daumen_beuger: int


@dataclass
class RigidBodyAssignment:
    zf_pp: int
    zf_dp: int
    daumen_dp: int
    daumen_mc: int
    force_torque: int


@dataclass
class JointInfo:
    name: str
    path: str


@dataclass
class AxisInfos:
    dau_cmc: JointInfo
    dau_mcp: JointInfo
    dau_pip: JointInfo
    dau_dip: JointInfo
    zf_mcp: JointInfo
    zf_pip: JointInfo
    zf_dip: JointInfo


@dataclass
class HandSettings:
    gen: GenSettings
    model_path: str
    mujoco_file: str
    scene_file: str
    optitrack_file: str
    gym_file: str
    use_gym: bool
    create_dataset: bool
    create_relative: bool
    create_video: bool
    dataset_name: str
    dataset_dict_name: str

    # thumb
    dau_dip: TrackSettings
    dau_pip: MarkerSettings
    dau_mcp: TrackSettings

    # index finger
    zf_dip: TrackSettings
    zf_mid1: MarkerSettings
    zf_mid2: MarkerSettings
    zf_mcp: TrackSettings

    # assignments
    motor_assign: FingerAssignment
    body_assign: RigidBodyAssignment

    # AxisNames
    axis_infos: AxisInfos
