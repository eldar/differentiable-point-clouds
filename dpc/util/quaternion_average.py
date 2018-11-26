"""
The MIT License (MIT)

Copyright (c) 2014 Tolga Birdal, Eldar Insafutdinov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Website: https://github.com/tolgabirdal/averaging_quaternions
"""

import numpy as np


def quatWAvgMarkley(Q):
    """
    ported from the original Matlab implementation at:
    https://www.mathworks.com/matlabcentral/fileexchange/40098-tolgabirdal-averaging_quaternions

    by Tolga Birdal
    Q is an Mx4 matrix of quaternions. weights is an Mx1 vector, a weight for
    each quaternion.
    Qavg is the weightedaverage quaternion
    This function is especially useful for example when clustering poses
    after a matching process. In such cases a form of weighting per rotation
    is available (e.g. number of votes), which can guide the trust towards a
    specific pose. weights might then be interpreted as the vector of votes
    per pose.
    Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
    "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30,
    no. 4 (2007): 1193-1197.
    """

    # Form the symmetric accumulator matrix
    A = np.zeros((4, 4))
    M = Q.shape[0]
    weights = np.ones(M)

    wSum = 0

    for i in range(M):
        q = Q[i, :]
        q = np.expand_dims(q, -1)
        w_i = weights[i]
        A = w_i * np.matmul(q, q.transpose()) + A  # rank 1 update
        wSum = wSum + w_i

    # scale
    A = 1.0 / wSum * A

    # Get the eigenvector corresponding to largest eigen value
    w, v = np.linalg.eig(A)
    ids = np.argsort(w)
    idx = ids[-1]
    q_avg = v[:, idx]
    if q_avg[0] < -0:
        q_avg *= -1.0
    return q_avg