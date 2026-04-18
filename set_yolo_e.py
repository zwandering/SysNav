from ultralytics import YOLOWorld, YOLO, YOLOE
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

# model = YOLOE("./src/semantic_mapping/semantic_mapping/external/yoloe-11l-seg.pt")
# model.set_classes(object_list, model.get_text_pe(object_list))

# if os.path.exists("./src/semantic_mapping/semantic_mapping/external/yoloe-11l-seg_cus.onnx"):
#     os.remove("./src/semantic_mapping/semantic_mapping/external/yoloe-11l-seg_cus.onnx")
#     print("Removed existing ONNX file.")
# if os.path.exists("./src/semantic_mapping/semantic_mapping/external/yoloe-11l-seg_cus.engine"):
#     os.remove("./src/semantic_mapping/semantic_mapping/external/yoloe-11l-seg_cus.engine")
#     print("Removed existing TensorRT engine file.")

model = YOLOE("./src/semantic_mapping/semantic_mapping/external/yoloe-26x-seg.pt")
model.set_classes(object_list, model.get_text_pe(object_list))

if os.path.exists("./src/semantic_mapping/semantic_mapping/external/yoloe-26x-seg_cus.onnx"):
    os.remove("./src/semantic_mapping/semantic_mapping/external/yoloe-26x-seg_cus.onnx")
    print("Removed existing ONNX file.")
if os.path.exists("./src/semantic_mapping/semantic_mapping/external/yoloe-26x-seg_cus.engine"):
    os.remove("./src/semantic_mapping/semantic_mapping/external/yoloe-26x-seg_cus.engine")
    print("Removed existing TensorRT engine file.")

engine_path = model.export(
    format="engine",   # TensorRT
    device=0,          # GPU
    imgsz=(640, 1920),
    half=True,         # FP16
    dynamic=False,
    workspace=6,        # TensorRT workspace (GB); 视显存而定，可 2~8
)
print("Exported:", engine_path)

