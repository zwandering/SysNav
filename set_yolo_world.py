from ultralytics import YOLOWorld, YOLO
import yaml
import os

object_file_path = 'src/semantic_mapping/semantic_mapping/config/objects.yaml'
with open(object_file_path, "r") as file:
    object_config = yaml.safe_load(file)
label_template = object_config['prompts']
object_list = []
for value in label_template.values():
    object_list += value['prompts']
print(f"Object List: {object_list}")

model = YOLOWorld("./src/semantic_mapping/semantic_mapping/external/yolov8x-worldv2.pt")
model.set_classes(object_list)
if os.path.exists("./src/semantic_mapping/semantic_mapping/external/yolov8x-worldv2_cus.pt"):
    os.remove("./src/semantic_mapping/semantic_mapping/external/yolov8x-worldv2_cus.pt")
    print("Removed existing custom model file.")
model.save("./src/semantic_mapping/semantic_mapping/external/yolov8x-worldv2_cus.pt")

model = YOLOWorld("./src/semantic_mapping/semantic_mapping/external/yolov8x-worldv2_cus.pt")

if os.path.exists("./src/semantic_mapping/semantic_mapping/external/yolov8x-worldv2_cus.onnx"):
    os.remove("./src/semantic_mapping/semantic_mapping/external/yolov8x-worldv2_cus.onnx")
    print("Removed existing ONNX file.")
if os.path.exists("./src/semantic_mapping/semantic_mapping/external/yolov8x-worldv2_cus.engine"):
    os.remove("./src/semantic_mapping/semantic_mapping/external/yolov8x-worldv2_cus.engine")
    print("Removed existing TensorRT engine file.")

engine_path = model.export(
    format="engine",   # TensorRT
    device=0,          # GPU
    imgsz=(640, 1920),
    half=True,         # FP16
    dynamic=False,
    workspace=4,        # TensorRT workspace (GB); 视显存而定，可 2~8
)
print("Exported:", engine_path)