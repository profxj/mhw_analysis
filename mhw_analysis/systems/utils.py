import numpy as np

def prep_labels(mask, parent, NSpax, MinNSpax=0, verbose=False):
    # !..this is the number of individual connected components found in the cube:
    nlabels = np.max(mask)
    nobj=np.sum(parent[1:nlabels+1]==0)
    if verbose:
        print("NObj Extracted=",nobj)

    LabelToId = np.zeros(nlabels+1, dtype=np.int32) -1
    IdToLabel = np.zeros(nobj, dtype=np.int32)

    #!----- DETECTION (using NSpax) -------------
    # !..build auxiliary arrays and count detections
    ndet=0
    for i in range(1,nlabels+1):
        if parent[i] == 0:
            this_label = i
            this_NSpax = NSpax[this_label] # 0-indexing
            if this_NSpax > MinNSpax:
                IdToLabel[ndet] = this_label
                LabelToId[this_label] = ndet
                ndet = ndet + 1  # ! update ndet
    if verbose:
        print('Nobj Detected =', ndet)

    # Return
    return IdToLabel, LabelToId, ndet
