from omegaconf import OmegaConf

class Parameter:
    def __init__(self, )

class Sampler:
    def __init__(self, config):
        config_dict = OmegaConf.to_container(cfg=config)
        hyper_params = config_dict['hyper']
        for param in hyper_params:
            
