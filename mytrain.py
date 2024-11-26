from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolov8-CBAM.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8-GAM.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8-CA.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8-SE.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8n.pt")  # 从头开始构建新模型
    # model.load('yolov8n.pt')  # 加载预训练模型（推荐用于训练）
    # Use the model
    results = model.train(data='F:/deep-learning-model/yolov8/ultralytics/data/qiaoliang.yaml',epochs=200,name='CBAM+WIoU')  # 训练模型
