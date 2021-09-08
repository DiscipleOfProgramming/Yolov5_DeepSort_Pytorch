import numpy as np
from dataclasses import dataclass

def xyxy_to_xy_centroid(xyxy):
    """Assumes tlhw format"""
    centroid_y = (xyxy[1] + xyxy[-1]) / 2
    centroid_x = (xyxy[0] + xyxy[-2]) / 2
    arr = np.array([centroid_x, centroid_y])
    return arr

def find_closest_box(id, other_ids, thresh, id_cent_dict):
    """Find the closest box to the missing box in the previous frame"""
    distances = [(np.linalg.norm(id_cent_dict[id] - id_cent_dict[o_id]), o_id) for o_id in other_ids]
    closest_dist_idx = min(distances, key=lambda d: d[0])
    if closest_dist_idx[0] < thresh:
        return closest_dist_idx[1]
    else:
        return None

def id_recovery(ids_in_last_frame: set, ids_curr_frame: set, cent_id_dict: dict, lost_id_list: list, frame_thresh=5):
    """Attempt at lost_id alg 2"""
    lost_ids = ids_in_last_frame.difference(ids_curr_frame)
    pot_new_ids = ids_curr_frame.difference(ids_in_last_frame)
    new_ids = [id for id in pot_new_ids if id not in cent_id_dict]
    # Take care of ids that have been lost by tracker/ yolov5
    # Remove ids that have been lost for too long
    for id, frames_present in lost_id_list:
        if frames_present > frame_thresh:
            lost_id_list.remove((id, frames_present))
    # Keep running list of ids that are lost
    map(lambda x: lost_id_list.append((x, 0)), new_ids)

@dataclass(frozen=True, eq=True)
class Id_pair:
    id: int
    closest_id: int

if __name__ == "__main__":
    xyxy_to_xy_centroid(np.array([803, 516, 873, 653]))