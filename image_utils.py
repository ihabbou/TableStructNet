import copy
import numpy as np


def image_size(path):
    from PIL import Image
    img = Image.open(path)
    return img.size

#########################################



def coords_to_bbox(coords):
    return coords[0], coords[1], \
                    coords[2] - coords[0], coords[3] - coords[1]

def bbox_to_coords(bbox): # TODO check
    return bbox[0], bbox[1], \
                    bbox[2] + bbox[0], bbox[3] + bbox[1]

def _centroid_from_coords(coords):
    return ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)

def _centroid_from_bbox(bbx):
    return (bbx[0]+bbx[2]/2, bbx[1]+bbx[3]/2)


def get_centroids_from_coords(coord_list):
    # TODO assert
    return [_centroid_from_coords(coords) for coords in coord_list]


def get_centroids_from_bboxes(bbox_list):
    # TODO assert
    return [_centroid_from_bbox(bbx) for bbx in bbox_list]


#########################################


def coords_in_table(coords, table_coords, intersection_threshold=0.9):
    return intersection_area(coords, table_coords) >= \
        ((coords[2] - coords[0])*(coords[3] - coords[1]))*intersection_threshold

def intersection_area(coorda, coordb):  # returns None if rectangles don't intersect
    dx = min(coorda[2], coordb[2]) - max(coorda[0], coordb[0])
    dy = min(coorda[3], coordb[3]) - max(coorda[1], coordb[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return -1

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

#########################################

#other
#https://stackoverflow.com/questions/55593506

#https://stackoverflow.com/questions/61874011
def merge_words(words, threshold_x=10, threshold_y=4):
    
    # merge pairs

    pairs = []

    threshold_y = threshold_y # height threshold
    threshold_x = threshold_x # x threshold

    for i in range(len(words)):
        for j in range(i+1, len(words)):

            x1i, y1i, x2i, y2i = words[i]['coords']
            x1j, y1j, x2j, y2j = words[j]['coords']
            # first of all, they should be in the same height range, starting Y axis should be almost same
            # their starting x axis is close upto a threshold

            cond1 = (abs(y1i - y1j) < threshold_y)
            cond2 = (abs(x2i - x1j) < threshold_x)
            cond3 = (abs(x2i - x1i) < threshold_x)

            if cond1 and (cond2 or cond3):
                pairs.append([i,j])

    # merging
    merged_pairs = []

    for i in range(len(pairs)):
        cur_set = set()
        p = pairs[i]

        done = False
        for k in range(len(merged_pairs)):
            if p[0] in merged_pairs[k]:
                merged_pairs[k].append(p[1])
                done = True
            if p[1] in merged_pairs[k]:
                merged_pairs[k].append(p[0])
                done = True

        if done:
            continue

        cur_set.add(p[0])
        cur_set.add(p[1])

        match_idx = []
        while True:
            num_match = 0
            for j in range(i+1, len(pairs)):
                p2 = pairs[j]

                if j not in match_idx and (p2[0] in cur_set or p2[1] in cur_set):
                    cur_set.add(p2[0])
                    cur_set.add(p2[1])
                    num_match += 1
                    match_idx.append(j)

            if num_match == 0:
                break
        merged_pairs.append(list(cur_set))

    merged_pairs = [list(set(a)) for a in merged_pairs]
    # add singles
    merged_pairs = merged_pairs + [[idx] 
                                    for idx in range(len(words)) 
                                    if idx not in [midx 
                                                    for mlis in merged_pairs
                                                    for midx in mlis]]
    #     print(merged_pairs)
    # alt merging
    # import networkx as nx
    # g = nx.Graph()
    # g.add_edges_from(pairs) # pass pairs here

    # gs = [list(a) for a in list(nx.connected_components(g))] # get merged pairs here
    # print(gs)


    # 
    out_final = []

    INF = 999999999 # a large number greater than any co-ordinate
    for idxs in merged_pairs:
        c_bbox = []

        for i in idxs:
            c_bbox.append(words[i])

        sorted_x = sorted(c_bbox, key =  lambda x: x['coords'][0])

        new_sol = {}
        new_sol['text'] = ''
        new_sol['coords'] = [INF, INF, -INF, -INF]
        new_sol['conf'] = np.mean([k['conf'] for k in sorted_x])
        for k in sorted_x:
            new_sol['text'] += ' ' + k['text']

            new_sol['coords'][0] = min(new_sol['coords'][0], k['coords'][0]) #x1
            new_sol['coords'][1] = min(new_sol['coords'][1], k['coords'][1]) #y1

            new_sol['coords'][2] = max(new_sol['coords'][2], k['coords'][2]) #x2
            new_sol['coords'][3] = max(new_sol['coords'][3], k['coords'][3]) #y2


        out_final.append(new_sol)
    return out_final


###################################


def shorten_tall_words(words, too_high_pctile=80, new_pctile=20):
    
    shortw = copy.deepcopy(words)
    
    heights = [word["coords"][3] - word["coords"][1]
                    for word in shortw]
    high_height = np.percentile(heights, too_high_pctile) #80% percentile
    #avg_height = np.median(heights)
    avg_height = np.percentile(heights, new_pctile)

    #print("avg", np.mean(heights))
    #print("med", np.median(heights))

    # make the box height no more than the 80 precentile
    for word in shortw:
        if (word["coords"][3] - word["coords"][1]) > high_height: 
            word["coords"][1] = (word["coords"][1] + word["coords"][3]) // 2 - avg_height // 2
            word["coords"][3] = (word["coords"][1] + word["coords"][3]) // 2 + avg_height // 2
            
    return shortw



###################################

def scale_centroids(centroid_list, tbounds):

    tx1, ty1, tx2, ty2 = tbounds
    
    c_norms = [(c[0] - tx1, c[1] - ty1) for c in centroid_list]
    tbounds_norm = tx1 - tx1, ty1 - ty1, tx2 - tx1, ty2 - ty1

    cs = [(c_norm[0]/tbounds_norm[2], c_norm[1]/tbounds_norm[3]) for c_norm in c_norms]
    return cs

