import importlib

from huggingface_hub import hf_hub_url
import requests
from io import BytesIO
from torch import load

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if not "params" in config or config["params"] is None:
        return get_obj_from_str(config["target"])()
    else:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))



def get_state_dict_from_hub(filename, repo_id="thuerey-group/p3d"):

    url = hf_hub_url(repo_id=repo_id, filename=filename)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    checkpoint = load(BytesIO(response.content), map_location="cpu", weights_only=True)
    state_dict = checkpoint['state_dict']

    # strip the 'model.' prefix from the state dict keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('model.', '', 1)
        new_state_dict[new_key] = value

    return new_state_dict