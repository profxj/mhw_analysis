""" Codes to visualize MHW systems"""
import numpy as np
from mayavi import mlab
from IPython import embed

def test_mayavi(cube, scale=0.25):
    #
    size = (1050, 1050)
    fig = mlab.figure(1, bgcolor=(0.5, 0.5, 0.5), size=size)

    #mlab.contour3d(cube, transparent=True, contours=[1.], opacity=0.7, vmin=1)

    # Volume
    source = mlab.pipeline.scalar_field(cube)
    mlab.pipeline.volume(source)

    # Axes
    mlab.axes(color=(1,1,1),
              extent=[1,cube.shape[0],0.,scale*cube.shape[1],0.,scale*cube.shape[2]],
              ylabel='lon', zlabel='lat', xlabel='day')
    mlab.show()
    #mlab.savefig('tst.png')

# Testing
if __name__ == '__main__':
    tst_file = '../../doc/nb/tst_cube.npy'
    cube = np.load(tst_file)
    test_mayavi(cube)
