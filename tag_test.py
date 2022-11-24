#!/usr/bin/python3

import sys
import cv2
import apriltag
import logging
import argparse
from pprint import pprint
import os.path
import re
from codetimer import CodeTimer
import numpy as np
import math
import dt_apriltags


# in inches
MARKER_SIZE_16 = 6.0
MARKER_SIZE_36 = 6.5
MARKER_HEIGHT_36 = 58.75
MARKER_X_24 = 21.0
MARKER_X_14 = 67.375


def marker_corners(c, size):
    halfsize = size / 2.0
    dy = np.array((0.0, halfsize, 0.0))
    dz = np.array((0.0, 0.0, halfsize))
    return np.array((c + dy - dz, c - dy - dz, c - dy + dz, c + dy + dz), dtype=np.float32)


def markers_and_board(family):
    '''Create a Board of markers matching the test pictures'''

    if family == 'tag36h11':
        tag_family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

        centers = {14: np.array((0.0, MARKER_X_14, MARKER_HEIGHT_36)),
                   24: np.array((0.0, MARKER_X_24, MARKER_HEIGHT_36)), }

        ids = tuple(centers.keys())
        corners = [marker_corners(centers[i], MARKER_SIZE_36) for i in ids]
    else:
        tag_family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)

        centers = {0: np.array((0.0, 50.25, 58.75)),
                   29: np.array((6.25, -6.0, 59.25)), }

        ids = tuple(centers.keys())
        corners = [marker_corners(centers[i], MARKER_SIZE_16) for i in ids]

    board = cv2.aruco.Board.create(corners, tag_family, ids)

    # print('board ids', board.getIds())
    # print('board corners', board.getObjPoints())
    return centers, board


_camera_matrix = None
distortion_coeff = np.array(5*[0.0, ])


def camera_matrix(shape):
    global _camera_matrix

    if _camera_matrix is None:
        fov_x = 64.0                  # degrees
        fx = shape[1] / (2.0 * math.tan(math.radians(fov_x) / 2.0))
        _camera_matrix = np.array([[fx, 0.0, shape[1]/2.0], [0.0, fx, shape[0]/2.0], [0.0, 0.0, 1.0]])
        # print('cam_matrix:', shape, _camera_matrix)
    return _camera_matrix


def camera_pose(tvec, rvec):
    rot, _ = cv2.Rodrigues(rvec)
    rot_inv = rot.transpose()

    # location of camera (0,0,0) in World coordinates
    x_w_r0 = -1 * np.matmul(rot_inv, tvec)

    # compute yaw angle (degrees)
    yaw = math.degrees(math.atan2(rot_inv[1][2], rot_inv[0][2]))

    return x_w_r0, yaw


def map_apriltag_to_opencv(apriltag_res):
    '''Transform the output from the AprilTag library to match OpenCV'''

    # output ids is a numpy vector of the ids
    # output corners is a list of numpy matrices, but the order changes
    # match the corner order to the output of OpenCV
    ordermap = (1, 0, 3, 2)

    ids = []
    corners = []
    for tag in apriltag_res:
        ids.append([tag['id']])
        t = tag['lb-rb-rt-lt']
        # careful: this is very heavily nested, lots of brackets
        corners.append(np.array([[t[i] for i in ordermap]], dtype=np.float32))
    return np.array(ids, dtype=np.int32), corners


def run_apriltag_lib(image_files, board, marker_centers,
                     output_dir=None, decimate=1.0, threads=1, refine_edges=True, maxhamming=0, family='tag36h11'):
    '''Use the official library'''

    found = 0

    # print('refine', refine_edges)
    detector = apriltag.apriltag(family, decimate=decimate, threads=threads, refine_edges=refine_edges, maxhamming=maxhamming)

    prev_rvec = None
    prev_tvec = None
    for fn in image_files:
        fn_base = os.path.basename(fn)
        with CodeTimer('image read'):
            file_img = cv2.imread(fn)

        with CodeTimer('grayscale'):
            gray_img = cv2.cvtColor(file_img, cv2.COLOR_BGR2GRAY)

        with CodeTimer('april detect'):
            results = detector.detect(gray_img)
            min_margin = min([r['margin'] for r in results]) if results else 0
            
            # for r in results:
            #     print(f"tag {r['id']}: hamming {r['hamming']} margin {r['margin']}")

        with CodeTimer('compute pose'):
            ids, corners = map_apriltag_to_opencv(results)

            found += len(ids)

            cam_matrix = camera_matrix(file_img.shape)

            # try fancy solvePnP to look for multiple solutions
            obj_pts, img_pts = cv2.aruco.getBoardObjectAndImagePoints(board, corners, ids)
            if obj_pts is not None:
                if True:
                    rc, rvec_list, tvec_list, rep_err = cv2.solvePnPGeneric(obj_pts, img_pts, cam_matrix, distortion_coeff,
                                                                            flags=cv2.SOLVEPNP_ITERATIVE,
                                                                            useExtrinsicGuess=(prev_rvec is not None),
                                                                            rvec=prev_rvec, tvec=prev_tvec)
                else:
                    # only works for a single tag, and produces 2 solutions
                    rc, rvec_list, tvec_list, rep_err = cv2.solvePnPGeneric(obj_pts, img_pts, cam_matrix, distortion_coeff,
                                                                            flags=cv2.SOLVEPNP_IPPE)

                for rv, tv, err in zip(rvec_list, tvec_list, rep_err):
                    cam_loc, yaw = camera_pose(tv, rv)
                    if output_dir:
                        pos = np.round(np.transpose(cam_loc)[0], 2)  # nicer to read
                        print(f'{fn_base} alltags: nused={len(obj_pts)/4} margin={min_margin:.1f} {pos} {yaw:.2f} reperr={err[0]:.3f}')
                    prev_rvec = rv
                    prev_tvec = tv
            elif output_dir:
                print(f'{fn_base} no tags identified')

            # --------------------------------
            # try each tag individually, using the full on as a starting point
            if False:
                # try again using the fancier solvePnP method
                for i in range(len(ids)):
                    # trans_vec1 = np.copy(trans_vec)
                    # rot_vec1 = np.copy(rot_vec)

                    obj_pts, img_pts = cv2.aruco.getBoardObjectAndImagePoints(board, corners[i], ids[i])
                    rc, rvec_list, tvec_list, rep_err = cv2.solvePnPGeneric(obj_pts, img_pts, cam_matrix, distortion_coeff, flags=cv2.SOLVEPNP_IPPE)
                    for rv, tv, err in zip(rvec_list, tvec_list, rep_err):
                        cam_loc, yaw = camera_pose(tv, rv)
                        if output_dir:
                            pos = np.round(np.transpose(cam_loc)[0], 2)  # nicer to read
                            print(f'{fn_base} tag  {ids[i][0]}: nused={nused} {pos} {yaw:.2f} reperr={err[0]:.3f}')

        if output_dir:
            # print()
            out_img = file_img.copy()
            cv2.aruco.drawDetectedMarkers(out_img, corners, ids)

            outfile = os.path.join(output_dir, fn_base)
            outfile = re.sub(r'\.jpg$', '.png', outfile, re.IGNORECASE)
            cv2.imwrite(outfile, out_img)

    return found


# DT crashes when it deletes the detector, so keep it forever
# Don't use this for production, but OK for testing
dt_detector = None


def run_dt_apriltags_lib(image_files, board, marker_centers,
                         output_dir=None, decimate=1.0, threads=1, refine_edges=True, maxhamming=1,
                         marker_size=MARKER_SIZE_16):
    '''Use the DT wrapper'''

    global dt_detector
    found = 0

    # print('refine', refine_edges)
    if dt_detector is None:
        dt_detector = dt_apriltags.Detector(families='tag36h11', quad_decimate=decimate, nthreads=threads, refine_edges=refine_edges)

    for fn in image_files:
        fn_base = os.path.basename(fn)

        with CodeTimer('image read'):
            file_img = cv2.imread(fn)

        cam_mat = camera_matrix(file_img.shape)
        cam_parm = [cam_mat[0][0], cam_mat[1][1], cam_mat[0][2], cam_mat[1][2]]

        with CodeTimer('grayscale'):
            gray_img = cv2.cvtColor(file_img, cv2.COLOR_BGR2GRAY)

        with CodeTimer('april detect'):
            results = dt_detector.detect(gray_img, estimate_tag_pose=True, camera_params=cam_parm,
                                         tag_size=marker_size)
            found += len(results)
            ids = np.array([[x.tag_id, ] for x in results], dtype=np.int32)
            corners = [np.array([x.corners, ], dtype=np.float32) for x in results]

        if output_dir:
            # print()
            out_img = file_img.copy()
            cv2.aruco.drawDetectedMarkers(out_img, corners, ids)

            outfile = os.path.join(output_dir, fn_base)
            outfile = re.sub(r'\.jpg$', '.png', outfile, re.IGNORECASE)
            cv2.imwrite(outfile, out_img)

    return found


def run_opencv_lib(image_files, board, marker_centers,
                   output_dir=None, decimate=1.0, threads=1, refine_edges=True, maxhamming=1, marker_size=MARKER_SIZE_16):
    '''Use the OpenCV Aruco library'''

    # version 4.6 does not seem to have the ArucoDetector class in Python

    tag_family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    det_params = cv2.aruco.DetectorParameters_create()
    det_params.aprilTagQuadDecimate = 1

    for fn in image_files:
        fn_base = os.path.basename(fn)

        with CodeTimer('image read'):
            file_img = cv2.imread(fn)

        with CodeTimer('grayscale'):
            gray_img = cv2.cvtColor(file_img, cv2.COLOR_BGR2GRAY)

        with CodeTimer('april detect'):
            corners, ids, _ = cv2.aruco.detectMarkers(gray_img, tag_family)
            print(repr(ids))
            print(repr(corners))
            print(len(ids), len(corners))

        cam_matrix = camera_matrix(file_img.shape)

        rvec_list, tvec_list, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, cam_matrix, distortion_coeff)
        for i in range(len(corners)):
            t = tvec_list[i][0]
            print(f'{fn_base} tag {ids[i][0]} ({t[2]:6.1f}, {t[0]:6.1f})')

        print(ids[1], corners[1], board.getObjPoints()[0])
        nused, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, distortion_coeff, None, None)
        print(f'{fn_base} all_tags ({tvec[0]}')

        if output_dir:
            out_img = file_img.copy()
            cv2.aruco.drawDetectedMarkers(out_img, corners, ids)

            outfile = os.path.join(output_dir, fn_base)
            outfile = re.sub(r'\.jpg$', '.png', outfile, re.IGNORECASE)
            cv2.imwrite(outfile, out_img)

    return len(ids)


def main():
    parser = argparse.ArgumentParser(description='Various tests of AprilTags')
    parser.add_argument('--output-dir', '-o', help='Output directory for processed images')
    parser.add_argument('--family', default='tag36h11', help='Tag family: tag16h5 or tag36h11')
    parser.add_argument('--decimate', default=1, type=float, help='Run many timing loops')
    parser.add_argument('--threads', default=1, type=int, help='Run many timing loops')
    parser.add_argument('--no-refine-edges', action='store_false', help='Run many timing loops')
    parser.add_argument('--time', type=int, default=1, help='Run many timing loops')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--processor', '-p', default='apriltag', help='Which processing library')
    parser.add_argument('files', nargs='+', help='Input files')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
    logging.getLogger().setLevel(logging.WARNING - 10 * args.verbose)

    if args.processor.lower() == 'apriltag':
        proc_func = run_apriltag_lib
    elif args.processor.lower() == 'opencv':
        proc_func = run_opencv_lib
    elif args.processor.lower() == 'dt_apriltags':
        proc_func = run_dt_apriltags_lib
    else:
        logging.error(f'Unknown processor "{args.processor}"')
        sys.exit(10)

    marker_centers, board = markers_and_board(args.family)

    nfound = 0
    if args.time > 1:
        for _ in range(args.time):
            nfound += proc_func(args.files, board, marker_centers,
                                decimate=args.decimate, threads=args.threads, refine_edges=args.no_refine_edges, family=args.family)
        CodeTimer.output_timers()
    else:
        nfound += proc_func(args.files, board, marker_centers, output_dir=args.output_dir,
                            decimate=args.decimate, threads=args.threads, refine_edges=args.no_refine_edges, family=args.family)

    print(f"Found {nfound/args.time} tags")
    return


if __name__ == '__main__':
    main()
