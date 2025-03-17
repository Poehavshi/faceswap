from math import acos, cos, degrees, sin

import cv2
import numpy as np

DTYPE = np.float32
CV2_INTERP = cv2.INTER_LINEAR


class UnknownShapeException(Exception):
    pass


def _transform_img(img, M, dsize, flags=CV2_INTERP, borderMode=None):
    """conduct similarity or affine transformation to the image, do not do border operation!
    img:
    M: 2x3 matrix or 3x3 matrix
    dsize: target shape (width, height)
    """
    _dsize = tuple(dsize) if isinstance(dsize, tuple | list) else (dsize, dsize)

    if borderMode is not None:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags, borderMode=borderMode, borderValue=(0, 0, 0))
    else:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)


def _transform_pts(pts, M):
    """conduct similarity or affine transformation to the pts
    pts: Nx2 ndarray
    M: 2x3 matrix or 3x3 matrix
    return: Nx2
    """
    return pts @ M[:2, :2].T + M[:2, 2]


def parse_pt2_from_pt101(pt101, use_lip=True):
    """
    parsing the 2 points according to the 101 points, which cancels the roll
    """
    # the former version use the eye center, but it is not robust, now use interpolation
    pt_left_eye = np.mean(pt101[[39, 42, 45, 48]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt101[[51, 54, 57, 60]], axis=0)  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt101[75] + pt101[81]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt106(pt106, use_lip=True):
    """
    parsing the 2 points according to the 106 points, which cancels the roll
    """
    pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt203(pt203, use_lip=True):
    """
    parsing the 2 points according to the 203 points, which cancels the roll
    """
    pt_left_eye = np.mean(pt203[[0, 6, 12, 18]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt203[[24, 30, 36, 42]], axis=0)  # right eye center
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt203[48] + pt203[66]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt68(pt68, use_lip=True):
    """
    parsing the 2 points according to the 68 points, which cancels the roll
    """
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55], dtype=np.int32) - 1
    if use_lip:
        pt5 = np.stack(
            [
                np.mean(pt68[lm_idx[[1, 2]], :], 0),  # left eye
                np.mean(pt68[lm_idx[[3, 4]], :], 0),  # right eye
                pt68[lm_idx[0], :],  # nose
                pt68[lm_idx[5], :],  # lip
                pt68[lm_idx[6], :],  # lip
            ],
            axis=0,
        )

        pt2 = np.stack([(pt5[0] + pt5[1]) / 2, (pt5[3] + pt5[4]) / 2], axis=0)
    else:
        pt2 = np.stack(
            [
                np.mean(pt68[lm_idx[[1, 2]], :], 0),  # left eye
                np.mean(pt68[lm_idx[[3, 4]], :], 0),  # right eye
            ],
            axis=0,
        )

    return pt2


def parse_pt2_from_pt5(pt5, use_lip=True):
    """
    parsing the 2 points according to the 5 points, which cancels the roll
    """
    if use_lip:
        pt2 = np.stack([(pt5[0] + pt5[1]) / 2, (pt5[3] + pt5[4]) / 2], axis=0)
    else:
        pt2 = np.stack([pt5[0], pt5[1]], axis=0)
    return pt2


def parse_pt2_from_pt9(pt9, use_lip=True):
    """
    parsing the 2 points according to the 9 points, which cancels the roll
    ['right eye right', 'right eye left', 'left eye right', 'left eye left', 'nose tip', 'lip right', 'lip left', 'upper lip', 'lower lip']
    """
    if use_lip:
        pt9 = np.stack(
            [
                (pt9[2] + pt9[3]) / 2,  # left eye
                (pt9[0] + pt9[1]) / 2,  # right eye
                pt9[4],
                (pt9[5] + pt9[6]) / 2,  # lip
            ],
            axis=0,
        )
        pt2 = np.stack([(pt9[0] + pt9[1]) / 2, pt9[3]], axis=0)  # eye  # lip
    else:
        pt2 = np.stack(
            [
                (pt9[2] + pt9[3]) / 2,
                (pt9[0] + pt9[1]) / 2,
            ],
            axis=0,
        )

    return pt2


def parse_pt2_from_pt478(pt478, use_lip=True):
    """
    parsing the 2 points according to the 101 points, which cancels the roll
    """
    # the former version use the eye center, but it is not robust, now use interpolation
    pt_left_eye = pt478[468]  # left eye center
    pt_right_eye = pt478[473]  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = pt478[14]
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt_x(pts, use_lip=True):
    if pts.shape[0] == 101:
        pt2 = parse_pt2_from_pt101(pts, use_lip=use_lip)
    elif pts.shape[0] == 106:
        pt2 = parse_pt2_from_pt106(pts, use_lip=use_lip)
    elif pts.shape[0] == 68:
        pt2 = parse_pt2_from_pt68(pts, use_lip=use_lip)
    elif pts.shape[0] == 5:
        pt2 = parse_pt2_from_pt5(pts, use_lip=use_lip)
    elif pts.shape[0] == 478:
        pt2 = parse_pt2_from_pt478(pts, use_lip=use_lip)
    elif pts.shape[0] == 203:
        pt2 = parse_pt2_from_pt203(pts, use_lip=use_lip)
    elif pts.shape[0] > 101:
        # take the first 101 points
        pt2 = parse_pt2_from_pt101(pts[:101], use_lip=use_lip)
    elif pts.shape[0] == 9:
        pt2 = parse_pt2_from_pt9(pts, use_lip=use_lip)
    else:
        raise UnknownShapeException()

    if not use_lip:
        # NOTE: to compile with the latter code, need to rotate the pt2 90 degrees clockwise manually
        v = pt2[1] - pt2[0]
        pt2[1, 0] = pt2[0, 0] - v[1]
        pt2[1, 1] = pt2[0, 1] + v[0]

    return pt2


def parse_rect_from_landmark(pts, scale=1.5, need_square=True, vx_ratio=0, vy_ratio=0, use_deg_flag=False, **kwargs):
    """parsing center, size, angle from 101/68/5/x landmarks
    vx_ratio: the offset ratio along the pupil axis x-axis, multiplied by size
    vy_ratio: the offset ratio along the pupil axis y-axis, multiplied by size, which is used to contain more forehead area

    judge with pts.shape
    """
    pt2 = parse_pt2_from_pt_x(pts, use_lip=kwargs.get("use_lip", True))

    uy = pt2[1] - pt2[0]
    length = np.linalg.norm(uy)
    if length <= 1e-3:
        uy = np.array([0, 1], dtype=DTYPE)
    else:
        uy /= length
    ux = np.array((uy[1], -uy[0]), dtype=DTYPE)

    # the rotation degree of the x-axis, the clockwise is positive, the counterclockwise is negative (image coordinate system)
    # print(uy)
    # print(ux)
    angle = acos(ux[0])
    if ux[1] < 0:
        angle = -angle

    # rotation matrix
    M = np.array([ux, uy])

    # calculate the size which contains the angle degree of the bbox, and the center
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T  # (M @ P.T).T = P @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2

    size = rb_pt - lt_pt
    if need_square:
        m = max(size[0], size[1])
        size[0] = m
        size[1] = m

    size *= scale  # scale size
    center = center0 + ux * center1[0] + uy * center1[1]  # counterclockwise rotation, equivalent to M.T @ center1.T
    center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)  # considering the offset in vx and vy direction

    if use_deg_flag:
        angle = degrees(angle)

    return center, size, angle


def _estimate_similar_transform_from_pts(pts, dsize, scale=1.5, vx_ratio=0, vy_ratio=-0.1, flag_do_rot=True, **kwargs):
    """calculate the affine matrix of the cropped image from sparse points, the original image to the cropped image, the inverse is the cropped image to the original image
    pts: landmark, 101 or 68 points or other points, Nx2
    scale: the larger scale factor, the smaller face ratio
    vx_ratio: x shift
    vy_ratio: y shift, the smaller the y shift, the lower the face region
    rot_flag: if it is true, conduct correction
    """
    center, size, angle = parse_rect_from_landmark(
        pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio, use_lip=kwargs.get("use_lip", True)
    )

    s = dsize / size[0]  # scale
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=DTYPE)  # center of dsize

    if flag_do_rot:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = center[0], center[1]  # ori center
        tcx, tcy = tgt_center[0], tgt_center[1]  # target center
        # need to infer
        M_INV = np.array(
            [
                [s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
                [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)],
            ],
            dtype=DTYPE,
        )
    else:
        M_INV = np.array([[s, 0, tgt_center[0] - s * center[0]], [0, s, tgt_center[1] - s * center[1]]], dtype=DTYPE)

    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)

    # M_INV is from the original image to the cropped image, M is from the cropped image to the original image
    return M_INV, M[:2, ...]


def crop_image(img, pts: np.ndarray, **kwargs):
    dsize = kwargs.get("dsize", 224)
    scale = kwargs.get("scale", 1.5)  # 1.5 | 1.6
    vy_ratio = kwargs.get("vy_ratio", -0.1)  # -0.0625 | -0.1

    M_INV, _ = _estimate_similar_transform_from_pts(
        pts,
        dsize=dsize,
        scale=scale,
        vy_ratio=vy_ratio,
        flag_do_rot=kwargs.get("flag_do_rot", True),
    )

    img_crop = _transform_img(img, M_INV, dsize)  # origin to crop
    pt_crop = _transform_pts(pts, M_INV)

    M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=DTYPE)])
    M_c2o = np.linalg.inv(M_o2c)

    ret_dct = {
        "M_o2c": M_o2c,  # from the original image to the cropped image 3x3
        "M_c2o": M_c2o,  # from the cropped image to the original image 3x3
        "img_crop": img_crop,  # the cropped image
        "pt_crop": pt_crop,  # the landmarks of the cropped image
    }

    return ret_dct
