"""Original native24 DINO-Large models with the existing freeze/LR policy."""
from .dino_web_model import DinoWebSegmentation
from .target_dino_model import configure_train_stage, parameter_groups
def build_model(config, *, load_pretrained=True):
    if config['classes'] != 24: raise ValueError('native24 classifier required')
    model=DinoWebSegmentation(variant=config['variant'],num_classes=24,pretrained=True,
        pretrained_file=config['pretrained_file'],pretrained_file_sha256=config['pretrained_sha256'],
        load_pretrained=load_pretrained,gradient_checkpointing=True,
        freeze_prefix_tokens=config['variant']=='v3l')
    configure_train_stage(model,0)
    return model
