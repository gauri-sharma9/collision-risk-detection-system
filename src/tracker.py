import math


class Tracker:
    def __init__(self):
        self.next_id        = 0
        self.objects        = {}
        self.previous       = {}
        self.missing_frames = {}
        self.max_missing    = 15   # keep objects longer during turns

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, detections):
        updated_objects = {}
        used_ids        = set()

        for det in detections:
            center       = self._get_center(det["bbox"])
            assigned_id  = None
            min_distance = float("inf")

            for obj_id, obj_center in self.objects.items():
                if obj_id in used_ids:
                    continue
                distance = math.dist(center, obj_center)
                # 200px threshold handles fast ego-vehicle motion
                if distance < min_distance and distance < 200:
                    min_distance = distance
                    assigned_id  = obj_id

            if assigned_id is None:
                assigned_id = self.next_id
                self.next_id += 1

            used_ids.add(assigned_id)
            self.previous[assigned_id]       = self.objects.get(assigned_id, center)
            updated_objects[assigned_id]     = center
            self.missing_frames[assigned_id] = 0
            det["id"]          = assigned_id
            det["displacement"] = math.dist(center, self.previous[assigned_id])

        for obj_id in list(self.objects.keys()):
            if obj_id not in updated_objects:
                self.missing_frames[obj_id] += 1
                if self.missing_frames[obj_id] <= self.max_missing:
                    updated_objects[obj_id] = self.objects[obj_id]
                else:
                    del self.missing_frames[obj_id]
                    self.previous.pop(obj_id, None)

        self.objects = updated_objects
        return detections