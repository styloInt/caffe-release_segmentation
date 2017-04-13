import caffe
import sys
sys.path.append('../')

# import surgery, score

import numpy as np
import os



try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('./models/bvlc_alexnet_FCN_wTOPO/solver_alexnet_fcn_softmaxloss.prototxt')
# solver.net.copy_from(weights)

# surgeries
# interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
# surgery.interp(solver.net, interp_layers)

# scoring
# val = np.loadtxt('../data/pascal/VOC2010/ImageSets/Main/val.txt', dtype=str)

for _ in range(50):
    solver.step(8000)
    score.seg_tests(solver, False, val, layer='score')
