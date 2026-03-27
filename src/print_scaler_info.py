import torch
s=torch.load('model_MT_196.pt', weights_only=False)
for k in ['scaler_y1','scaler_y4']:
    sc=s[k]
    print(k)
    print('min_', sc.min_)
    print('scale_', sc.scale_)
    print('data_min_', sc.data_min_)
    print('data_max_', sc.data_max_)
