import cv2
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

# ---- config ----
WEIGHTS_PATH = "hotdog_weights.pth"
CLASS_NAMES = ["hotdog","not_hotdog" ] 
NUM_CLASSES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- model ----
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

state_dict = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()
print("Loaded model + weights, set to eval.")

# ---- preprocessing ----
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])


transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

@torch.no_grad()
def predict_frame(bgr_frame):
    # BGR (OpenCV) -> RGB (PIL)
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    x = transform(pil_img).unsqueeze(0).to(device)  # (1,3,224,224)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]         # (num_classes,)
    pred_idx = int(torch.argmax(probs).item())
    conf = float(probs[pred_idx].item())

    return pred_idx, conf, probs.cpu().tolist()

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pred_idx, conf, _ = predict_frame(frame)
        label = f"{CLASS_NAMES[pred_idx]} ({conf:.2f})"

        # FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hotdog Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
