import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU training is not recommended
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Use a specific GPU id, e.g. "0" or "0,1"
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18-soep-p3p4.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=8,
                workers=4,
                seed=0,
                # device='0,1',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )
