#!/usr/bin/python3

import sys
import cv2
import apriltag
import logging
import argparse
# from pprint import pprint
import os.path
import re
from codetimer import CodeTimer
import numpy as np
import math


# in inches
MARKER_LENGTH = 6.5
MARKER_HEIGHT = 58.75
MARKER_X_24 = 21.0
MARKER_X_14 = 67.375


def marker_corners(c, l):
    l2 = l / 2.0
    return ((0.0, c[0] + l2, c[1] - l2), (0.0, c[0] - l2, c[1] - l2), (0.0, c[0] - l2, c[1] + l2), (0.0, c[0] + l2, c[1] + l2))


def create_board():
    '''Create a Board of markers matching the test pictures'''

    tag_family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    corners = []
    corners.append(np.array(marker_corners((MARKER_X_14, MARKER_HEIGHT), MARKER_LENGTH), dtype=np.float32))
    corners.append(np.array(marker_corners((MARKER_X_24, MARKER_HEIGHT), MARKER_LENGTH), dtype=np.float32))
    # print('corners', repr(corners))

    ids = (14, 24)

    board = cv2.aruco.Board.create(corners, tag_family, ids)

    print('board ids', board.getIds())
    print('board corners', board.getObjPoints())
    return board


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


def run_apriltag_lib(image_files, camera_matrix, distortion_coeff, board,
                     output_dir=None, decimate=1.0, threads=1, refine_edges=True, maxhamming=1):
    '''Use the official library'''

    found = 0

    # print('refine', refine_edges)
    detector = apriltag.apriltag('tag36h11', decimate=decimate, threads=threads, refine_edges=refine_edges, maxhamming=maxhamming)
    for fn in image_files:
        fn_base = os.path.basename(fn)
        with CodeTimer('image read'):
            file_img = cv2.imread(fn)

        with CodeTimer('grayscale'):
            gray_img = cv2.cvtColor(file_img, cv2.COLOR_BGR2GRAY)

        with CodeTimer('april detect'):
            results = detector.detect(gray_img)
            # pprint(results)

        ids, corners = map_apriltag_to_opencv(results)
        found += len(ids)

        rvec_list, tvec_list, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, distortion_coeff)
        for i in range(len(corners)):
            t = tvec_list[i][0]
            print(f'{fn_base} tag {ids[i][0]} ({t[2]:6.1f}, {t[0]:6.1f})')

        #print(ids[0], corners[0], board.getObjPoints()[0])
        nused, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, distortion_coeff, None, None)
        print(f'{fn_base} all_tags {np.transpose(tvec)[0]}')

        if output_dir:
            out_img = file_img.copy()
            cv2.aruco.drawDetectedMarkers(out_img, corners, ids)

            outfile = os.path.join(output_dir, fn_base)
            outfile = re.sub(r'\.jpg$', '.png', outfile, re.IGNORECASE)
            cv2.imwrite(outfile, out_img)

    return found


def run_opencv_lib(image_files, camera_matrix, distortion_coeff, board,
                   output_dir=None, decimate=1.0, threads=1, refine_edges=True, maxhamming=1):
    '''Use the OpenCV Aruco library'''

    # version 4.6 does not seem to have the ArucoDetector class in Python

    tag_family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # det_params = cv2.aruco.DetectorParameters()
    # det_params.aprilTagQuadDecimate = 1

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

        rvec_list, tvec_list, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, distortion_coeff)
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
    parser.add_argument('--decimate', default=1, type=float, help='Run many timing loops')
    parser.add_argument('--threads', default=1, type=int, help='Run many timing loops')
    parser.add_argument('--no-refine-edges', action='store_false', help='Run many timing loops')
    parser.add_argument('--time', type=int, help='Run many timing loops')
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
    else:
        logging.error(f'Unknown processor "{args.processor}"')
        sys.exit(10)

    xsize = 640.0
    ysize = 480.0
    fov_x = 64.0                  # degrees
    fx = xsize / (2.0 * math.tan(math.radians(fov_x) / 2.0))
    camera_matrix = np.array([[fx, 0.0, xsize/2.0], [0.0, fx, ysize/2.0], [0.0, 0.0, 1.0]])
    distortion_coeff = np.array(5*[0.0, ])
    # print("cam", camera_matrix, "dist", distortion_coeff)

    board = create_board()
    
    nfound = 0
    if args.time:
        for _ in range(args.time):
            nfound += proc_func(args.files, camera_matrix, distortion_coeff, board, decimate=args.decimate,
                                threads=args.threads, refine_edges=args.no_refine_edges)
        CodeTimer.output_timers()
    else:
        nfound += proc_func(args.files, camera_matrix, distortion_coeff, board, output_dir=args.output_dir,
                            decimate=args.decimate, threads=args.threads, refine_edges=args.no_refine_edges)

    print(f"Found {nfound} tags")
    return


if __name__ == '__main__':
    main()
