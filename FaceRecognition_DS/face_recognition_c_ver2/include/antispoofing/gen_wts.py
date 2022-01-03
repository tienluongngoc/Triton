import torch
import struct
import sys
import torch
sys.path.insert(0, "Detection")

# Initialize
device = torch.device('cuda')
pt_file = sys.argv[1]
# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

with open(pt_file.split('.')[0] + '.wts', 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')