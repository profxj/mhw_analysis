""" Module to generate MHW systems

Some of the code that follows is from the routine “Extract.f90” in CubEx by S. Cantalupo.
That’s the core of the code. The inputs are the Datacube, another cube for the
flagging criteria (the Variance for astronomical cubes) and the output is a 3D segmentation map
or mask (called “Mask” or “mask”). The routine WriteCheckCubes.f90 is where this 3D Mask is used
for various sort of outputs.

Please note that for any astronomical purposes, the LICENCE in the folder applies
(this includes any modification of the original code).   Here it is:

Copyright: Sebastiano Cantalupo 2019.

CubEx (and associated software in the folder "Tools" and "Scripts") non-public software licence notes.

This is a non-public software:

1) you cannot distribute any part of this software without the authorization by the copyright holder.

2) you can modify any part of the software, however, any modifications do not change the copyright 
    and distribution rights stated in the point 1 above.

3) publications using results obtained with any part of this software in the original of modified form
    as stated in point 2, should include the copyright owner in the author list unless otherwise stated by
    the copyright owner.  

4) downloading, opening, installing and/or using the package is equivalent to accepting points 1, 2 and 3 above.
"""
import os
import numpy as np

from mhw_analysis.systems import utils


from IPython import embed


def define_systems(cube, verbose=True, MinNSpax=0, return_first=False,
                   return_second=False):
    """


    Args:
        cube:
        verbose:
        MinNSpax:

    Returns:

    """

    #!..make sure that undefined variance pixels that are defined in the datacube have a dummy high value
    #WHERE(Cube/=UNDEF.and.(Var==UNDEF.or.Var==0.)) Var=1.e30
    #bad = np.where(Cube != np.nan  and (Var==np.nan | Var==0.))[0]
    #Var[bad] = 1.e30

    label = 0
    mask = np.zeros_like(cube, dtype=np.int32)
    parent = np.zeros(10000000, dtype=np.int32)
    NSpax = np.zeros_like(parent)
    maxnlabels = parent.size

    DimX, DimY, DimZ = cube.shape
    # Loop me!
    for i in range(1, DimX - 1):
        for j in range(1,DimY-1):
            for k in range(1, DimZ - 1):

                #print(i,j,k)
                #IF(Cube(i, j, k) == UNDEF) CYCLE
                if not cube[i,j,k]:
                    continue

                #!..check (prior) neighbors of this spaxel, for simplicity we actually check ALL neighbors here
                #prior_labels = RESHAPE(mask(i-1:i+1,j-1:j+1,k-1:k+1),(/27/))
                prior_labels = mask[i-1:i+2,j-1:j+2,k-1:k+2].flatten()
                #import pdb; pdb.set_trace()

                if np.all(prior_labels == 0):
                    label=label+1
                    if label > maxnlabels: # STOP "Increase stack size (maxnlabels)!"
                       raise ValueError("Increase stack size for labels!")
                    mask[i,j,k] = label
                #ELSE !..this spaxel is connected to another one
                else: # !..this spaxel is connected to another one
                    #this_label = MINVAL(prior_labels, MASK=prior_labels /= 0)
                    this_label = np.min(prior_labels[prior_labels > 0])
                    if this_label == 0:
                        embed(header='bug on 73')
                    mask[i,j,k] = this_label
                    #!..update parent tree
                    #DO p = 1, SIZE(prior_labels)
                    for p in range(prior_labels.size):
                        if prior_labels[p] != 0 and prior_labels[p] != this_label:
                            #CALL union(this_label, prior_labels(p))
                            union(parent, this_label, prior_labels[p])

    if return_first:
        return mask, parent

    # !..second pass:
    # !... replace labels using the parent tree
    # !... get NSpax for each individual connected component
    #  DO k=1,DimZ
    #     DO j=1,DimY
    #        DO i=1,DimX
    for i in range(DimX):
        for j in range(DimY):
            for k in range(DimZ):
                this_label=mask[i,j,k]
                if this_label != 0:
                    #!..assign value from parent tree
                    p = this_label
                    while parent[p] != 0:
                       p = parent[p]

                    mask[i,j,k] = p
    #               !..update NSpax counter associated with this label
                    NSpax[p] = NSpax[p]+1

    if return_second:
        return mask, parent, NSpax

    '''
    nlabels = np.max(mask)
    # !..this is the number of individual connected components found in the cube:
    nobj=np.sum(parent[1:nlabels+1]==0)
    if verbose:
        print("NObj Extracted=",nobj)

    # Allocate
    LabelToId = np.zeros(nlabels+1, dtype='int') -1
    IdToLabel = np.zeros(nobj, dtype='int')
    

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
    '''

    IdToLabel, LabelToId, ndet = utils.prep_labels(mask, parent, NSpax, MinNSpax=MinNSpax, verbose=verbose)
    # Objects
    obj_dict = dict(Id=np.zeros(ndet, dtype=np.int32), NSpax=np.zeros(ndet, dtype=np.int32), # Assoc=[0]*ndet,
                    xcen=np.zeros(ndet, dtype=np.float32), xboxmin=np.ones(ndet, dtype=np.int32)*1e5, xboxmax=np.ones(ndet, dtype=np.int32)*-1,
                    ycen=np.zeros(ndet, dtype=np.float32), yboxmin=np.ones(ndet, dtype=np.int32)*1e5, yboxmax=np.ones(ndet, dtype=np.int32)*-1,
                    zcen=np.zeros(ndet, dtype=np.float32), zboxmin=np.ones(ndet, dtype=np.int32)*1e5, zboxmax=np.ones(ndet, dtype=np.int32)*-1)

    # Init
    for ii in range(ndet):
        obj_dict['Id'][ii] = ii+1
        obj_dict['NSpax'][ii] = NSpax[IdToLabel[ii]]

    # Fill !..find bounding boxes and centroid for each objects
    for i in range(DimX):
        for j in range(DimY):
            for k in range(DimZ):

                this_label = mask[i, j, k]
                if this_label != 0:
                    id = LabelToId[this_label]  #!..get object associated with pixel  (0-based)
                    if id != -1:
                        obj_dict['xcen'][id] += i - 0.5
                        obj_dict['ycen'][id] += j - 0.5
                        obj_dict['zcen'][id] += k - 0.5

                        obj_dict['xboxmin'][id] = min(obj_dict['xboxmin'][id], i)
                        obj_dict['yboxmin'][id] = min(obj_dict['yboxmin'][id], j)
                        obj_dict['zboxmin'][id] = min(obj_dict['zboxmin'][id], k)

                        obj_dict['xboxmax'][id] = max(obj_dict['xboxmax'][id], i)
                        obj_dict['yboxmax'][id] = max(obj_dict['yboxmax'][id], j)
                        obj_dict['zboxmax'][id] = max(obj_dict['zboxmax'][id], k)
                    else: # Cleanup mask
                        mask[i, j, k] = 0


    # !..finalize geometrical centroid calculation
    for id in range(ndet):
        obj_dict['xcen'][id] /= obj_dict['NSpax'][id]
        obj_dict['ycen'][id] /= obj_dict['NSpax'][id]
        obj_dict['zcen'][id] /= obj_dict['NSpax'][id]

    # Finish
    # Return
    return mask, obj_dict

def union(parent, x, y):
    """

    Args:
       parent:
       x:
       y:

    Returns:

    """
    #!..find root labels
    while parent[x] != 0:
       x = parent[x]

    while parent[y] != 0:
       y = parent[y]

    if x < y:
       parent[y] = x
    elif x > y:
       parent[x] = y


