import yaml
from .get_root_dir import get_root_dir


def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.

    For example,
    [UCR] `yaml_fname`: "./examples/configs/example_ucr_vibcreg.yaml"
    [PTB-XL] `yaml_fname`: "./examples/configs/example_ptbxl_vibcreg.yaml"
    """
    stream = open(yaml_fname, 'r')
    cf = yaml.load(stream, Loader=yaml.FullLoader)  # config
    return cf
