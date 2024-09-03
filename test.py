import os
import torch
from models.blip_vqa import BLIP_VQA
import yaml

save_dir = '/output/VQA/lr2e-5_ep10/checkpoint_02.pth'
state_dict = torch.load(save_dir)['model']
# print(state_dict.keys())
config_dir = '/home/serendi/Desktop/Projects/BLIP-main/configs/rsvqa.yaml'

config = yaml.load(open(config_dir, 'r'), Loader=yaml.Loader)
model = BLIP_VQA(image_size = config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
model.load_state_dict(state_dict)

for name, parms in model.named_parameters():
# 	print('-->name:', name, '-->grad_requirs:', parms.requires_grad, 
#        '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
	# print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
 	print('-->name:', name, '-->grad_requirs:', parms.requires_grad, 
       '--weight', parms.data, ' -->grad_value:', parms.grad)