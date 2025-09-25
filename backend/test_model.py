import os, sys, torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from model import load_model, infer_instances

# ---- CONFIG ----
IMAGE_PATH = r"C:\Users\hardi\OneDrive\Desktop\hardik\college\BE\final year project\image-defect-detection\image-defect-detection\sample_images\133.jpg"
THRESH = float(os.getenv("DEFECT_CONF_THRESH", "0.3"))

# ---- LOAD MODEL ----
print("Loading model...")
model = load_model()

# ---- LOAD IMAGE ----
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image not found: {IMAGE_PATH}")
    sys.exit(1)

img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ---- RUN INFERENCE ----
print("Running inference...")
res = infer_instances(model, img_rgb)
boxes, scores, masks = res["boxes"], res["scores"], res["masks"]

print(f"Found {len(boxes)} instances above threshold {THRESH}")

# ---- VISUALIZE ----
plt.figure(figsize=(8,6))
plt.imshow(img_rgb)
ax = plt.gca()

for i, (box, score) in enumerate(zip(boxes, scores)):
    x1, y1, x2, y2 = box.astype(int)
    ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1,
                               fill=False, color='lime', linewidth=2))
    ax.text(x1, y1-2, f"{score:.2f}", color='yellow', fontsize=8,
            bbox=dict(facecolor='black', alpha=0.5, pad=1))
    if i < len(masks):
        cnts, _ = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            c = c.squeeze(1)
            plt.plot(c[:,0], c[:,1], color='red', linewidth=1)

plt.axis("off")
plt.title(f"Detections: {len(boxes)}")
plt.show()
