""" Codes to visualize MHW systems"""
import numpy as np
from mayavi import mlab
from IPython import embed

from mhw_analysis.systems import io as mhw_sys_io

def test_mayavi(cube, outfile='tst.png', scale=0.25, volume=False):
    #
    #embed(header='8 of viz')
    cntrs = np.unique(cube)[1:]
    size = (1050, 1050)
    fig = mlab.figure(1, bgcolor=(0.5, 0.5, 0.5), size=size)

    if not volume:
        mlab.contour3d(cube, transparent=True,
                       contours=cntrs.tolist(),
                       opacity=0.7, vmin=1)
    else: # Volume
        source = mlab.pipeline.scalar_field(cube)
        mlab.pipeline.volume(source)

    # Axes
    mlab.axes(color=(1,1,1),
              extent=[1,cube.shape[0],0.,scale*cube.shape[1],0.,scale*cube.shape[2]],
              ylabel='lon', zlabel='lat', xlabel='day')
    #mlab.show()
    mlab.savefig(outfile)

# Testing
if __name__ == '__main__':
    #tst_file = '../../doc/nb/tst_cube.npy'
    #tst_file = './tst_mask.npy'
    tst_file = './pacific_mask.npy'
    cube = np.load(tst_file)

    test_mayavi(cube)
