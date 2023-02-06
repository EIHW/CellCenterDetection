# -*- coding: utf-8 -*-
 
from typing import List
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def find_best_unique_match(
    S_1: List[Tuple[int,int]], # usually list of ground truth/reference locations
    S_2: List[Tuple[int,int]], # usually list of detected locations
    max_dist_threshold=7**2    # if more than this squared apart, two locations do not match
):                             # distance is squared to avoid taking sqrt()
    N_1 = len(S_1)
    N_2 = len(S_2)

    dist = np.zeros((N_1, N_2), dtype=np.float32)

    for i in range(0, N_1):
        for j in range(0, N_2):
            dx = S_1[i][0] - S_2[j][0]
            dy = S_1[i][1] - S_2[j][1]
            dist[i,j] = dx * dx + dy * dy

    min_N = min(N_1, N_2)
    R = []
    for _ in range(0, min_N):
        idx = np.unravel_index(dist.argmin(), dist.shape)
        v_min = dist[idx]
        if v_min > max_dist_threshold:
            # found match spatial too far away
            break
        R.append((idx[0], idx[1], v_min))
        dist[idx[0], :] = np.finfo(dist.dtype).max
        dist[:, idx[1]] = np.finfo(dist.dtype).max
        #print(idx)

    R = sorted(R, key=lambda x: x[2], reverse=False)
    return R

# -----------------------------------------------------------------------------
def determine_hits(
    S_1: List[Tuple[int,int,float]], # usually list of detected locations with confidence scores
    S_2: List[Tuple[int,int]], # usually list of ground truth/reference locations
    max_dist_threshold=7    # if more than this squared apart, two locations do not match
):                  
    max_dist_threshold = max_dist_threshold**2       # distance is squared to avoid taking sqrt()
    N_1 = len(S_1)

    N_2 = len(S_2)
    tp = 0
    fp = 0
    fn = 0

    if N_1 > 5* N_2:
        x=1 

    if N_1 <= 0:
        return [], tp, fp, fn

    S_1_ext = [x+[0,1] for x in S_1] 
    dist = np.zeros((N_1, N_2), dtype=np.float32)

    for i in range(0, N_1):
        for j in range(0, N_2):
            dx = S_1[i][0] - S_2[j][0]
            dy = S_1[i][1] - S_2[j][1]
            dist[i,j] = dx * dx + dy * dy

    min_N = min(N_1, N_2)
    
    distance_cap_constance = np.finfo(dist.dtype).max
    for _ in range(0, min_N):
        idx = np.unravel_index(dist.argmin(), dist.shape)
        v_min = dist[idx]
        if v_min > max_dist_threshold:
            # found match spatial too far away
            break
        # Set hit (index == 3) and fp (index == 4) indicator
        if len(S_1_ext[idx[0]]) < 5:
            S_1_ext[idx[0]][2] = 1
            S_1_ext[idx[0]][3] = 0
        else:
            S_1_ext[idx[0]][3] = 1
            S_1_ext[idx[0]][4] = 0

        dist[idx[0], :] = distance_cap_constance
        dist[:, idx[1]] = distance_cap_constance
        # every entry update is one true positive
        tp += 1
    # every prediction that has not been assigned to a gt box is a false positive
    fp = N_1 - tp
    # gt box that has not been assigned to a prediction is a false negative
    fn = N_2 - tp
    return S_1_ext, tp, fp, fn # (pos 1, pos 2, confidence, hit, fp), true positives, false positives, false negatives



def compute_precision_recall_values(R, N_1, N_2, max_dist_threshold=3**2):
    prec_recs = []
    len_R =len(R)
    if len_R == 0:
        if N_2 > 0:
            # N_1 == 0
            prec_recs.append([0.0,1.0])
        else:
            # N_2 == 0
            prec_recs.append([1.0,0.0])
        return prec_recs

    prec_recs.append([1.0,0.0])
    n = 0
    while True:
        if not (n < len_R):
            break
        x,y, v_min = R[n]
        thres = v_min
        if thres > max_dist_threshold:
            break
        while (n < len_R - 1) and R[n+1][2] <= thres:
            n += 1

        hits = n+1
        misses = N_1 - hits
        false_pos = N_2 - hits

        precision = hits / N_2 # N_2 = (hits + false_pos)
        recall = hits / N_1 # N_1 = (hits + misses)
        prec_recs.append([precision,recall])

        n += 1

    # add dummy for mAP calculation
    prec_recs.append([0,1.0])
    return prec_recs


def plot_prec_rec(title: str, prec_recs):
    data = np.array(prec_recs)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.grid()
    plt.title(title)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.plot(data[:,1], data[:,0])
    plt.show()


# Similar to VOC2011 challenge ap calculation
def compute_AP(
    rec: List[float],
    prec:List[float]
):
    mrec = np.array([0] + rec + [1])
    mpre = np.array([0] + prec + [0])
    for i in range(mpre.shape[0]-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    # comes back as unmodifiable tuple. There convert to np.array
    diffs = mrec[1:] - mrec[0:-1]
    ap = np.sum( diffs * mpre[1:] )
    return ap
    

if __name__ == "__main__":
    S_1 = [(2,3), (3,6), (6,8)]
    S_2 = [(1,1), (2,2), (0,0), (0,1)]

    R = find_best_unique_match(S_1, S_2)
    print ('R:', R)

    prec_recs = compute_precision_recall_values(R, len(S_1), len(S_2))
    print('prec_recs',prec_recs)

    plot_prec_rec("Precision Recall Curve", prec_recs)

    rec = [0.25, 0.5, 0.75]
    prec = [1.0, 0.8, 0.0]

    ap = compute_AP(rec, prec)