import numpy as np

def compass_angle(v: np.ndarray) -> float:
    """
    Takes a vector, and returns the compass angle in radians.
    0.0 at North, increasing clockwise (e.g. 90° at East).
    """
    return np.arctan2(v[0], v[1]) % (2 * np.pi)


def _test_compass_angle():
    # replace with print statements
    print(compass_angle(np.array([0, 1]))) # 0.0 (North)
    print(compass_angle(np.array([1, 0]))) # 90° (East)
    print(compass_angle(np.array([0, -1]))) # 180° (South)
    print(compass_angle(np.array([-1, 0]))) # 270° (West)

def _alpha(w: np.ndarray) -> float:
    """
    Takes the wind vector, and returns the alpha degree.
    If w is pointing southwards: alpha = 0.0. 
    Increases cw (90 pointing west, 180 pointing north, 270 pointing east).
    """
    return compass_angle(-w)
    #return np.arctan2(-w[0] , -w[1]) % (2 * np.pi)

def _test_alpha():
    # replace with print statements
    print(_alpha(np.array([0, -1]))) # 0.0 (wind points south)
    print(_alpha(np.array([-1, 0]))) # 90° (wind points west)
    print(_alpha(np.array([0, 1]))) # 180° (wind points north)
    print(_alpha(np.array([1, 0]))) # 270° (wind points east)

def _rotation(angle: float) -> np.ndarray:
    """2D rotation matrix for a given angle.
    Positive angle corresponds to clockwise rotation (e.g. 90° rotates north to west).
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])

def _test_rotation():
    # replace with print statements
    print(_alpha(_rotation(0) @ np.array([0, -1]))) # [1, 0]
    print(_alpha(_rotation(np.pi / 2) @ np.array([0, -1]))) # [0, 1]
    print(_alpha(_rotation(np.pi) @ np.array([0, -1]))) # [-1, 0]
    print(_alpha(_rotation(-np.pi / 2) @ np.array([0, -1]))) # [0, -1]


if __name__ == "__main__":
    _test_compass_angle()
    _test_alpha()
    _test_rotation()


