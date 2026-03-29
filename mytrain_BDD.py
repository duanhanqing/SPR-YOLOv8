import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/home/luoluo/work4/YOLOv8.2/configs3/2YOLOv8_ACmix_CSPStage.yaml')
    # model.load(r'/szzn/luoluo/work2/YOLOv8.2/yolov8n.pt')
    model.train(
                data=r'/home/luoluo/work4/YOLOv8.2/ultralytics/cfg/datasets/BDD_100K.yaml',
                imgsz=640,
                epochs=300,
                batch=128,
                patience=0,
                workers=64,
                device="4,5,6,7",
                optimizer='SGD',  
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project= '/home/luoluo/work4/YOLOv8.2/runs3/BDD_100K/2YOLOv8_ACmix_CSPStage/train', 
            )
    
#    export PYTHONPATH=$PYTHONPATH:/home/luoluo/work4/YOLOv8.2
#    /opt/anaconda3/envs/luoluo/bin/python /home/luoluo/work4/YOLOv8.2/mytrain_BDD.py