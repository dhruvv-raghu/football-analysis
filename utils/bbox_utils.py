def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_width_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int(x2 - x1)

def measure_distance(x1, x2):
    return ((x1[0]- x2[0])**2 + (x1[1] - x2[1])**2)**0.5