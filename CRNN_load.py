class values:
  num_class = 36
  Prediction = 'Attn'
  image_folder = '/home/number_plates'
  saved_model = '/home/crnn_weights/content/deep-text-recognition-benchmark/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth'
  workers = 4
  batch_size = 192
  batch_max_length = 25
  imgH = 32
  imgW = 100
  rgb = False
  character = '0123456789abcdefghijklmnopqrstuvwxyz'
  sensitive = False
  PAD = True
  Transformation = 'TPS'
  FeatureExtraction = 'ResNet'
  SequenceModeling = 'BiLSTM'
  Prediction = 'Attn'
  num_fiducial = 20
  input_channel = 1
  output_channel = 512
  hidden_size = 256
  
  
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import time


from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#device = torch.device('cpu')
opt = values()
converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)
model = Model(opt)
print('model input parameters', 32, 100, 20, 3, 512,
          256, 36, 25, 'TPS', 'ResNet',
          'BiLSTM', 'Attn')
model = torch.nn.DataParallel(model).to(device)

# load model
path_of_model = '/home/crnn_weights/deep-text-recognition-benchmark/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth'
print('loading pretrained model from %s' % opt.saved_model)
num_workers=int(opt.workers)

if opt.sensitive:
  opt.character = string.printable[:-6]
cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()

model.load_state_dict(torch.load(opt.saved_model, map_location=device))

start_time = time.time()
AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
demo_data = RawDataset(root=opt.image_folder, opt=opt)
demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
 
for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

  
preds = model(image, text_for_pred, is_train=False)

_, preds_index = preds.max(2)
preds_str = converter.decode(preds_index, length_for_pred)
#print(preds_str)
preds_prob = F.softmax(preds, dim=2)
preds_max_prob, _ = preds_prob.max(dim=2)
for pred in preds_str:
  pred_EOS = pred.find('[s]')
  pred = pred[:pred_EOS]  
  preds_max_prob = preds_max_prob[:pred_EOS]
  #print(pred)
print(time.time()-start_time)
