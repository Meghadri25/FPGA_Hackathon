import torch
import numpy as np

state = torch.load('model_MT_196.pt', weights_only=False)
model_state = state['model_state_dict']

np.savez('model_params.npz',
         backbone_0_weight=model_state['backbone.0.weight'].cpu().numpy(),
         backbone_0_bias=model_state['backbone.0.bias'].cpu().numpy(),
         backbone_3_weight=model_state['backbone.3.weight'].cpu().numpy(),
         backbone_3_bias=model_state['backbone.3.bias'].cpu().numpy(),
         backbone_6_weight=model_state['backbone.6.weight'].cpu().numpy(),
         backbone_6_bias=model_state['backbone.6.bias'].cpu().numpy(),
         head_h1_weight=model_state['head_h1.weight'].cpu().numpy(),
         head_h1_bias=model_state['head_h1.bias'].cpu().numpy(),
         head_h4_weight=model_state['head_h4.weight'].cpu().numpy(),
         head_h4_bias=model_state['head_h4.bias'].cpu().numpy(),
        )
print('Saved model_params.npz')