"""Contact sheet of the depth benchmark's novel-view splats (ground truth | DA2 | DA3 metric | DA3 posed | DrivingForward)."""
import glob
import os

import cv2
import numpy as np

from realityview import paths

D = os.path.join(paths.OUTPUTS, "depth_benchmark")
names = ["gt", "dav2", "da3metric", "da3_pose12", "df_gs"]
rows = []
for g in sorted(glob.glob(f"{D}/s*_T*_gt.jpg"))[:4]:
    b = g[:-len("gt.jpg")]
    rows.append(np.hstack([cv2.putText(cv2.resize(cv2.imread(b + n + ".jpg"), (400, 220)), n, (6, 20), 0, 0.6, (0, 255, 255), 2) for n in names]))
cv2.imwrite(f"{D}/sheet.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("wrote", f"{D}/sheet.jpg")
