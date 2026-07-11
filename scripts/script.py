from roboflow import Roboflow

rf = Roboflow(api_key="EyR2nq8F7lqbnnIyKzW4")
project = rf.workspace("pramanarendra").project("web-ui-element-detection")
version = project.version(1)

# Download to a specific folder
dataset = version.download(
    model_format="yolov8",
    location="./datasets/web-ui-detection"  # 👈 Your specified path
)

print(f"Dataset downloaded to: {dataset.location}")