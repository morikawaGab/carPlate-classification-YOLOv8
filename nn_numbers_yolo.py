from ultralytics import YOLO
import numpy as np
import os

TRAIN = True
VALIDATE = False

def trainModel(dataset_path):
    model = YOLO('yolov8n-cls.pt')  # carrega um modelo pré-treinado
    model.train(data=dataset_path, epochs=100, imgsz=240)

def validateModel():
    path_to_best = './runs/classify/train/weights/best.pt'
    model = YOLO(path_to_best)

    metrics = model.val()
    metrics.top1   # top1 acurácia
    metrics.top5   # top5 acurácia

def main():

    dataset_path = os.path.abspath(os.getcwd()) + '/characterDataset/numbers'

    if TRAIN: trainModel(dataset_path)
    if VALIDATE: validateModel()
    #if TEST: testModel(dataset_path)

if __name__ == '__main__':
    main()