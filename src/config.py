DOTA_CLASSES = ["plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle", "ship", "tennis-court",
                "basketball-court", "storage-tank", "soccer-ball-field", "roundabout", "harbor", "swimming-pool", "helicopter"]
OUTPUT_STRIDE = 4

CLASS_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
    (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(DOTA_CLASSES)}
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GT_YELLOW = (255, 255, 0)