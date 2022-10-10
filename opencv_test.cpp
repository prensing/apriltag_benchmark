#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <cstdlib>
#include <string>
#include <time.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
  struct timespec startt, endt;
  int repetitions = 100;
  
  Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_APRILTAG_36h11);
  Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
  parameters->aprilTagQuadDecimate = 2.0;
  //parameters->useAruco3Detection = true;
  //parameters->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
  //parameters->cornerRefinementMethod = aruco::CORNER_REFINE_CONTOUR;
  //parameters->cornerRefinementMethod = aruco::CORNER_REFINE_APRILTAG;
  
  cout << "aruco3 " << parameters->useAruco3Detection << endl;
  cout << "refine " << parameters->cornerRefinementMethod << endl;
  cout << "decimate " << parameters->aprilTagQuadDecimate << endl;
  cout << "sigma " << parameters->aprilTagQuadSigma << endl;

  double detector_time = 0.0;
  int detector_count = 0;
  int found = 0;
  
  for (int r=0; r< repetitions; r++) {
    for (int i=1; i<argc; i++) {
      // cout << argv[1] << endl;
      Mat in_image = imread(argv[i], IMREAD_COLOR);
      
      Mat gray_image;
      cvtColor(in_image, gray_image, COLOR_BGR2GRAY);

      clock_gettime(CLOCK_MONOTONIC, &startt);
    
      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f> > corners, rejectedCandidates;
      aruco::detectMarkers(gray_image, dictionary, corners, ids, parameters, rejectedCandidates);

      clock_gettime(CLOCK_MONOTONIC, &endt);
      detector_time += (endt.tv_sec - startt.tv_sec) * 1000.0 + (endt.tv_nsec - startt.tv_nsec) / 1.0e6;
      detector_count++;

      found += ids.size();
      if (repetitions <= 1) {
        Mat out_image;
        in_image.copyTo(out_image);
        //cout << "Found " << ids.size() << " markers" << endl;
        aruco::drawDetectedMarkers(out_image, corners, ids);
        string outname = "output/" + std::to_string(i) + ".png";
        // cout << "writing to " << outname << endl;
        imwrite(outname, out_image);
      }
    }
  }

  cout << "tags found = " << found << endl;
  cout << "Num images = " << detector_count << " total time = " << detector_time
       << "  time/frame (ms) = " << detector_time / detector_count << endl;
}
