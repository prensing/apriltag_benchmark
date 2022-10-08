#!/usr/bin/python3

import cv2
import apriltag
import logging
import argparse
from pprint import pprint
import os.path
import re
from codetimer import CodeTimer

TAG_FAMILY = 'tag36h11'


def highlight_tag(image, box, id):
    '''Draw a box around the tag and display the ID'''

    fontsize = image.shape[0] / 1000
    if image.shape[0] < 400:
        thickness = 1
    elif image.shape[0] < 700:
        thickness = 2
    else:
        thickness = 4

    pts = box.astype(int)
    cv2.polylines(image, [pts], 1, (0, 255, 0), thickness)
    id_coord = (pts[2][0] + 10, pts[2][1] - 10)
    cv2.putText(image, str(id), id_coord, cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), thickness=thickness)
    return


def run_apriltag_lib(image_files, output_dir=None):
    '''Use the official library'''

    detector = apriltag.apriltag(TAG_FAMILY)
    for fn in image_files:
        with CodeTimer('image read'):
            file_img = cv2.imread(fn)

        with CodeTimer('grayscale'):
            gray_img = cv2.cvtColor(file_img, cv2.COLOR_BGR2GRAY)

        with CodeTimer('april detect'):
            results = detector.detect(gray_img)
            #pprint(results)

        if output_dir:
            out_img = file_img.copy()
            for tag in results:
                highlight_tag(out_img, tag['lb-rb-rt-lt'], tag['id'])

            outfile = os.path.join(output_dir, os.path.basename(fn))
            outfile = re.sub(r'\.jpg$', '.png', outfile, re.IGNORECASE)
            cv2.imwrite(outfile, out_img)

    return


def main():
    parser = argparse.ArgumentParser(description='Various tests of AprilTags')
    parser.add_argument('--output-dir', '-o', help='Output directory for processed images')
    parser.add_argument('--time', type=int, help='Run many timing loops')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('files', nargs='+', help='Input files')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
    logging.getLogger().setLevel(logging.WARNING - 10 * args.verbose)

    if args.time:
        for _ in range(args.time):
            run_apriltag_lib(args.files)
        CodeTimer.output_timers()
    else:
        run_apriltag_lib(args.files, args.output_dir)

    return


if __name__ == '__main__':
    main()
