import numpy as np


class CMC:
    # Modified from TrackTrack: the original reads precomputed warp matrices of the MOT17/MOT20/DanceTrack
    # videos from files. The aquarium cameras are static, so no camera motion is compensated.
    def __init__(self, vid_name):
        super(CMC, self).__init__()

    def get_warp_matrix(self):
        return np.eye(2, 3, dtype=np.float64)


def apply_cmc(tracks, warp_matrix=np.eye(2, 3)):
    # Check
    if len(tracks) == 0:
        return 0

    # Get mean, covariance
    multi_mean = np.asarray([t.mean.copy() for t in tracks])
    multi_covariance = np.asarray([t.covariance for t in tracks])

    # Get warp matrix
    rot = warp_matrix[:, :2]
    rot_8x8 = np.kron(np.eye(4, dtype=float), rot)
    trans = warp_matrix[:, 2]

    # Warp
    for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        mean = rot_8x8 @ mean
        mean[:2] += trans
        cov = rot_8x8 @ cov @ rot_8x8.T

        tracks[i].mean = mean
        tracks[i].covariance = cov
