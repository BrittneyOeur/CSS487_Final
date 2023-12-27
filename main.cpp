// main.cpp
// Demonstrates various methods to identify DS/3DS chips of the Animal Crossing series
// Author: Brittney Oeur

#include <iostream>
#include <filesystem>
#include <algorithm>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp> // Include SIFT header

using namespace std;
namespace fs = std::filesystem;
using namespace cv;

// loadImages - loads image paths from a directory
// Preconditions: directoryPath is a valid directory
// Postconditions: imagePaths contains paths of .jpg and .png files in the directory
void loadImages(const string& directoryPath, vector<string>& imagePaths) {
    // Check if the directory exists and is indeed a directory
    if (fs::exists(directoryPath) && fs::is_directory(directoryPath)) {
        // Iterate through each entry in the directory
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            // Check if the entry is a .jpg or .png file
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                // Add the path of the image file to the imagePaths vector
                imagePaths.push_back(entry.path().string());
            }
        }
    }
}

// createHistogram - generates a 3D color histogram for the input image
// Preconditions: thisImage is a valid image matrix (OpenCV Mat type)
// Postconditions: Returns a 3D histogram Mat of color distribution
Mat createHistogram(const Mat& thisImage) {
    // The # of buckets in each dimension
    const int size = 4;
    int bucketSize = 256 / size;

    // Store to track the highest count in the histogram
    int highestCount = 0; 

    // Store the most common color components
    int cRed;
    int cGreen;
    int cBlue;

    // Checks if the image exist or has failed to load
    if (thisImage.empty()) {
        cout << "Empty descriptors." << endl;
    }

    // Create an array of the histogram dimensions
    int dims[] = { size, size, size };

    // Create 3D histogram of integers initialized to zero	
    Mat hist(3, dims, CV_32S, Scalar::all(0));

    // Creates a 3D color histogram
    for (int y = 0; y < thisImage.rows; y++) {
        for (int x = 0; x < thisImage.cols; x++) {
            // Loops through each pixel in the foreground image
            Vec3b pixel = thisImage.at<Vec3b>(y, x);

            int red = pixel[2];
            int green = pixel[1];
            int blue = pixel[0];

            // Calculate the bin indices for the histogram
            int r = red / bucketSize;
            int g = green / bucketSize;
            int b = blue / bucketSize;

            // Increment the corresponding bin to the histogram
            hist.at<int>(r, g, b)++;
        }
    }
    // Return the generated 3D histogram
    return hist;
}

// mostCommonColor - identifies the approximate most common color from the histogram
// Preconditions: size is the number of buckets in each dimension, hist is a valid 3D histogram matrix
// Postconditions: Returns a Vec3b representing the most common color in the histogram
Vec3b mostCommonColor(const int& size, const Mat& hist) {
    // Track the highest count in the histogram
    int highestCount = 0;

    // Size of each bucket in the histogram
    int bucketSize = 256 / size;

    // Store the most common color components
    int cRed = 0;
    int cGreen = 0;
    int cBlue = 0;

    // Finds the most common color in the foreground image
    // looping over all three dimensions
    for (int r = 0; r < size; r++) {
        for (int g = 0; g < size; g++) {
            for (int b = 0; b < size; b++) {
                int count = hist.at<int>(r, g, b);

                // Counts for biggest bin, if the current
                // count is higher, then it updates the 'highestCount'
                if (count > highestCount) {
                    highestCount = count;

                    // Calculate the approx. most common color based on the bin indices
                    cRed = r * bucketSize + bucketSize / 2;
                    cGreen = g * bucketSize + bucketSize / 2;
                    cBlue = b * bucketSize + bucketSize / 2;

                }
            }
        }
    }
    return Vec3b(cBlue, cGreen, cRed);
}

// replaceForegroundWithColor - replaces the most common color in the image with a replacement color
// Preconditions: scene is a valid image matrix (OpenCV Mat type), replacementColor is a Vec3b representing the replacement color
// Postconditions: Modifies the scene by replacing pixels of the most common color with the replacement color
Mat replaceForegroundWithColor(const Mat& objectImage, const Vec3b& replacementColor) {
    // Create a copy of the input image to modify
    Mat modifiedImage = objectImage.clone();

    // Compute histogram for the scene image
    Mat hist = createHistogram(modifiedImage);

    // Find the most common color in the object image
    Vec3b mostCommon = mostCommonColor(4, hist);

    // Iterate through pixels in the modified image
    for (int y = 0; y < modifiedImage.rows; y++) {
        for (int x = 0; x < modifiedImage.cols; x++) {
            // Get the color of the current pixel
            Vec3b color = modifiedImage.at<Vec3b>(y, x);

            // Compare the current pixel's color with the most common color in the scene
            if (color == mostCommon) {
                // Replace the pixel color with the replacement color
                modifiedImage.at<Vec3b>(y, x) = replacementColor;
            }
        }
    }
    // Return the modified image
    return modifiedImage;
}

// gaussianBlur - blurs an image
// preconditions: image is an greyscale or color byte image
// postconditions: a new image with the same size is returned after blurring
Mat gaussianBlur(const Mat& image) {
    Mat result = image.clone();
    int kernelSize = 5;
    GaussianBlur(result, result, Size(kernelSize, kernelSize), 0, 0);
    return result;
}

// BORROWED FROM PREVIOUS PROGRAMMING ASSIGNMENT
// sharpen - sharpens an image by taking a weighted average of each pixels with its neighbors
// preconditions: image is an greyscale or color byte image
// postconditions: a new image with the same size is returned after sharpening
Mat sharpen(const Mat& image) {
    Mat result = image.clone();

    // If greyscale, handle single band
    if (image.channels() == 1) {
        for (int r = 1; r < image.rows - 1; r++) {
            for (int c = 1; c < image.cols - 1; c++) {
                result.at<uchar>(r, c) = saturate_cast<uchar>(5 * image.at<uchar>(r, c)
                    - image.at<uchar>(r + 1, c) - image.at<uchar>(r, c + 1)
                    - image.at<uchar>(r - 1, c) - image.at<uchar>(r, c - 1));
            }
        }
    }

    // If color, handle three bands
    if (image.channels() == 3) {
        for (int r = 1; r < image.rows - 1; r++) {
            for (int c = 1; c < image.cols - 1; c++) {
                for (int b = 0; b < 3; b++) {
                    result.at<Vec3b>(r, c)[b] = saturate_cast<uchar>(5 * image.at<Vec3b>(r, c)[b]
                        - image.at<Vec3b>(r + 1, c)[b] - image.at<Vec3b>(r, c + 1)[b]
                        - image.at<Vec3b>(r - 1, c)[b] - image.at<Vec3b>(r, c - 1)[b]);
                }
            }
        }
    }
    return result;
}

// COOccurrenceMatrix - computes the co-occurrence matrix from an input image
// preconditions: image is a single-channel (greyscale) image
//                 distance is the offset for comparing pixel intensities
//                 numLevels is the number of intensity levels
// postconditions: returns a normalized co-occurrence matrix
Mat COOccurenceMatrix(const Mat& image, int distance, int numLevels) {
    Mat coOccurrenceMatrix = Mat::zeros(numLevels, numLevels, CV_32F);

    // Iterate through the image pixels to compute the matrix
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols - distance; ++x) {
            int intensity1 = static_cast<int>(image.at<uchar>(y, x));
            int intensity2 = static_cast<int>(image.at<uchar>(y, x + distance));

            // Increment the co-occurrence counts for the corresponding intensities
            coOccurrenceMatrix.at<float>(intensity1, intensity2) += 1.0;
            coOccurrenceMatrix.at<float>(intensity2, intensity1) += 1.0;
        }
    }

    // Normalize the matrix
    normalize(coOccurrenceMatrix, coOccurrenceMatrix, 1.0, 0.0, NORM_L1);

    // Return the computed co - occurrence matrix
    return coOccurrenceMatrix;
}

// compareCoOccurrenceMatrices - compares two co-occurrence matrices using template matching
// preconditions: matrix1 and matrix2 are matrices of the same size and type
// postconditions: returns the correlation value between the two matrices
double compareCoOccurrenceMatrices(const Mat& matrix1, const Mat& matrix2) {
    // Initialize matrix to store correlation result
    Mat correlation;

    // Perform template matching using normalized cross-correlation
    matchTemplate(matrix1, matrix2, correlation, TM_CCORR_NORMED);

    // Get the correlation value at the top-left corner
    double correlationValue = correlation.at<float>(0, 0);

    // Return the computed correlation value
    return correlationValue;
}

// siftMatching - performs SIFT feature matching between an object image and a scene image
// preconditions: objectImage and sceneImage are valid input images
// postconditions: identifies keypoints and descriptors in both images and matches descriptors to find correspondences
void siftMatching(const Mat& objectImage, const Mat& sceneImage) {
    // Creating a pointer to a Feature2D object and initializing it with a SIFT
    Ptr<Feature2D> detector = SIFT::create();

    // Declaring empty vectors to store keypoints 
    // (locations in the image where the algorithm finds something unique)
    vector<KeyPoint> keypointsObject;
    vector<KeyPoint> keypointsScene;

    // Declaring empty Mats to store descriptors 
    // (vectors that represent the keypoints' characteristics)
    Mat descriptorsObject;
    Mat descriptorsScene;

    // Using the SIFT detector to find keypoints and compute their descriptors for both objectImage and sceneImage
    detector->detectAndCompute(objectImage, Mat(), keypointsObject, descriptorsObject);
    detector->detectAndCompute(sceneImage, Mat(), keypointsScene, descriptorsScene);

    // Checking if either the descriptors for the object or scene image are empty
    if (descriptorsObject.empty() || descriptorsScene.empty()) {
        cout << "Empty descriptors." << endl;
    }

    else {
        // Creating a Brute-Force Matcher object using L2 norm (Euclidean distance) for matching descriptors
        BFMatcher matcher(NORM_L2);
        vector<vector<DMatch>> matches;
        matcher.knnMatch(descriptorsObject, descriptorsScene, matches, 2);

        // Filter matches based on Lowe's ratio test
        vector<DMatch> goodMatches;
        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i][0].distance < 0.75 * matches[i][1].distance) {
                goodMatches.push_back(matches[i][0]);
            }
        }
        // Draw matching keypoints if there are good matches
        if (goodMatches.size() > 10) {
            vector<Point2f> obj;
            vector<Point2f> scene;

            // Collect keypoints from good matches for object and scene images
            for (size_t i = 0; i < goodMatches.size(); i++) {
                obj.push_back(keypointsObject[goodMatches[i].queryIdx].pt);
                scene.push_back(keypointsScene[goodMatches[i].trainIdx].pt);
            }

            // Find the homography between object and scene keypoints
            Mat H = findHomography(obj, scene, RANSAC);

            // Apply the homography transformation to object corners
            vector<Point2f> objCorners(4);
            objCorners[0] = Point2f(0, 0);
            objCorners[1] = Point2f(objectImage.cols, 0);
            objCorners[2] = Point2f(objectImage.cols, objectImage.rows);
            objCorners[3] = Point2f(0, objectImage.rows);

            // Transform object corners to scene perspective
            vector<Point2f> sceneCorners(4);
            perspectiveTransform(objCorners, sceneCorners, H);

            /*
            // Find the most common color in the object image
            Mat hist = createHistogram(objectImage);
            Vec3b mostCommon = mostCommonColor(4, hist);

            // Create a mask to represent the area around the detected object
            Mat mask = Mat::zeros(sceneImage.rows, sceneImage.cols, CV_8UC1); // Initialize a black mask

            vector<Point> polyCorners; // Create a single vector of points to represent the polygon

            // Convert Point2f to Point for fillPoly
            for (int i = 0; i < sceneCorners.size(); ++i) {
                polyCorners.push_back(Point(sceneCorners[i].x, sceneCorners[i].y));
            }

            vector<vector<Point>> contours;
            contours.push_back(polyCorners); // Store the polygon corners as the contour

            // Fill the mask with a white polygon representing the detected object
            fillPoly(mask, contours, Scalar(255));

            // Invert the mask to represent the background
            bitwise_not(mask, mask);

            // Replace the background of the found object in sceneImage with the most common color
            Mat resultScene = sceneImage.clone(); // Create a copy of sceneImage
            resultScene.setTo(Scalar(mostCommon[0], mostCommon[1], mostCommon[2]), mask);
            */

            // Draw matching keypoints
            Mat imgMatches;
            drawMatches(objectImage, keypointsObject, sceneImage, keypointsScene, goodMatches, imgMatches, Scalar::all(-1),
                 Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            //Draw the lines between the matched points (object) in the scene image
             for (int i = 0; i < 4; i++) {
                 line(imgMatches, sceneCorners[i % 4] + Point2f(objectImage.cols, 0),
                     sceneCorners[(i + 1) % 4] + Point2f(objectImage.cols, 0), Scalar(0, 255, 0), 4);
             }

            imshow("Detected Objects", imgMatches);
            // imshow("Detected Objects", resultScene);
            waitKey(0);
        }

        // Not enough good matches of keypoints
        else {
            cout << "No good matches found." << endl;
        }
    }
}

// orbMatching -- performs ORB feature matching between an object image and a scene image
// Preconditions: objectImage and sceneImage are valid input images
// Postconditions: detects ORB keypoints and computes descriptors for both images,
//                 matches descriptors to identify corresponding features
void orbMatching(const Mat& objectImage, const Mat& sceneImage) {
    Ptr<Feature2D> detector = ORB::create(
        500, // Number of features to detect
        1.2f, // Scale factor between levels in the scale pyramid
        8, // Number of pyramid levels
        31, // Edge threshold
        0, // The level of the pyramid to put the image
        2, // Number of points that produce each element of the oriented BRIEF descriptor
        ORB::HARRIS_SCORE, // Harris corner score used to rank points
        31, // Size of the patch used by the oriented BRIEF descriptor
        20 // Fast threshold
    );

    vector<KeyPoint> keypointsObject, keypointsScene;
    Mat descriptorsObject, descriptorsScene;

    detector->detectAndCompute(objectImage, Mat(), keypointsObject, descriptorsObject);
    detector->detectAndCompute(sceneImage, Mat(), keypointsScene, descriptorsScene);

    // Checking if either the descriptors for the object or scene image are empty
    if (descriptorsObject.empty() || descriptorsScene.empty()) {
        cout << "empty" << endl;
    }

    else {
        // Create a Brute-Force Matcher using Hamming distance
        BFMatcher matcher(NORM_HAMMING);
        vector<vector<DMatch>> knnMatches;

        // Perform KNN matching between object and scene descriptors
        matcher.knnMatch(descriptorsObject, descriptorsScene, knnMatches, 2);

        // Filter matches based on Lowe's ratio test
        vector<DMatch> goodMatches;
        for (size_t i = 0; i < knnMatches.size(); ++i) {
            if (knnMatches[i][0].distance < 0.75 * knnMatches[i][1].distance) {
                goodMatches.push_back(knnMatches[i][0]);
            }
        }

        // Draw matching keypoints if there are good matches
        if (goodMatches.size() > 10) {
            vector<Point2f> obj;
            vector<Point2f> scene;

            // Collect matching keypoints from good matches
            for (size_t i = 0; i < goodMatches.size(); i++) {
                obj.push_back(keypointsObject[goodMatches[i].queryIdx].pt);
                scene.push_back(keypointsScene[goodMatches[i].trainIdx].pt);
            }

            // Find the homography between object and scene keypoints
            Mat H = findHomography(obj, scene, RANSAC);

            // Apply the homography transformation to object corners
            vector<Point2f> objCorners(4);
            objCorners[0] = Point2f(0, 0);
            objCorners[1] = Point2f(objectImage.cols, 0);
            objCorners[2] = Point2f(objectImage.cols, objectImage.rows);
            objCorners[3] = Point2f(0, objectImage.rows);

            vector<Point2f> sceneCorners(4);
            perspectiveTransform(objCorners, sceneCorners, H);

            // Draw matching keypoints
            Mat imgMatches;
            drawMatches(objectImage, keypointsObject, sceneImage, keypointsScene, goodMatches, imgMatches, Scalar::all(-1),
                Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            //Draw the lines between the matched points (object) in the scene image
            for (int i = 0; i < 4; i++) {
                line(imgMatches, sceneCorners[i % 4] + Point2f(objectImage.cols, 0),
                    sceneCorners[(i + 1) % 4] + Point2f(objectImage.cols, 0), Scalar(0, 255, 0), 4);
            }

            imshow("Detected Objects", imgMatches);
            waitKey(0);
        }
    }
}

// templateMatching - performs template matching using the chosen method to find the object in the scene
// Preconditions: objectImage and sceneImage are valid input images
// Postconditions: computes the similarity between the object and scene using the chosen method
void templateMatching(const Mat& objectImage, Mat& sceneImage) {
    Mat result;
    int matchMethod = TM_CCOEFF_NORMED;

    // Apply template matching
    matchTemplate(sceneImage, objectImage, result, matchMethod);

    // Localize the best match with minMaxLoc
    double minVal;
    double maxVal;
    Point minLocation;
    Point maxLocation;
    minMaxLoc(result, &minVal, &maxVal, &minLocation, &maxLocation, Mat());

    // Define a region of interest (ROI) based on the template match
    Rect roiRect(maxLocation.x, maxLocation.y, objectImage.cols, objectImage.rows);

    Mat roi = sceneImage(roiRect).clone();

    siftMatching(objectImage, roi);
}

// main - starting point of the program
// Preconditions: Requires valid paths to directories containing images for two categories (ACWW and ACNL)
// Postconditions: Processes images from both directories, performs image matching, computes texture correlation,
//                 and displays visual results and correlation values for each image pair in the directories
int main(int argc, char* argv[]) {
    // Image portion
    vector<string> imagePathsWW;

    // Retrieve the path of the images
    string directoryPathWW = R"(.\acww_train)";

    loadImages(directoryPathWW, imagePathsWW);

    // Iterates through the directory
    for (const auto& imagePath : imagePathsWW) {
        Mat sceneImage = imread(imagePath, IMREAD_COLOR);
        Mat objectImage = imread("./acww/acww.jpg", IMREAD_COLOR);

        // Click 'Enter' to iterate through the images
        if (!objectImage.empty() && !sceneImage.empty()) {
            imshow("Scene Image", sceneImage);
            waitKey(0);

            Mat blurImage = gaussianBlur(sceneImage);
            imshow("Blurred Scene Image", blurImage);
            waitKey(0);

            siftMatching(objectImage, blurImage);
            // siftMatching(objectImage, sceneImage);
            // orbMatching(objectImage, blurImage);

        }

        else {
            cout << "ERROR: Failed to load images." << endl;
        }

        // Compute co-occurrence matrices for both images
        int distance = 1; // Distance between pixel pairs
        int numLevels = 256; // Number of gray levels (adjust as needed)
        Mat sceneCoOccurrenceMatrix = COOccurenceMatrix(sceneImage, distance, numLevels);
        Mat objectCoOccurrenceMatrix = COOccurenceMatrix(objectImage, distance, numLevels);

        // Compare the co-occurrence matrices using correlation
        double correlation = compareCoOccurrenceMatrices(sceneCoOccurrenceMatrix, objectCoOccurrenceMatrix);

        // Display the correlation value
        cout << "Correlation between sceneImage and objectImage textures: " << correlation << endl;
    }

    // Image portion
    vector<string> imagePathsNL;

    // Retrieve the path of the images
    string directoryPathNL = R"(.\acnl_train)";

    loadImages(directoryPathNL, imagePathsNL);

    // Iterates through the directory
    for (const auto& imagePath : imagePathsNL) {
        Mat sceneImage = imread(imagePath, IMREAD_COLOR);
        Mat objectImage = imread("./acnl/acnl.jpg", IMREAD_COLOR);

        // Click 'Enter' to iterate through the images
        if (!objectImage.empty() && !sceneImage.empty()) {
            imshow("Scene Image", sceneImage);
            waitKey(0);

            Mat blurImage = gaussianBlur(sceneImage);
            imshow("Blurred Scene Image", blurImage);
            waitKey(0);

            siftMatching(objectImage, blurImage);
            // siftMatching(objectImage, sceneImage);
        }

        else {
            cout << "ERROR: Failed to load images." << endl;
        }

        // Compute co-occurrence matrices for both images
        int distance = 1; // Distance between pixel pairs
        int numLevels = 256; // Number of gray levels (adjust as needed)
        Mat sceneCoOccurrenceMatrix = COOccurenceMatrix(sceneImage, distance, numLevels);
        Mat objectCoOccurrenceMatrix = COOccurenceMatrix(objectImage, distance, numLevels);

        // Compare the co-occurrence matrices using correlation
        double correlation = compareCoOccurrenceMatrices(sceneCoOccurrenceMatrix, objectCoOccurrenceMatrix);

        // Display the correlation value
        cout << "Correlation between sceneImage and objectImage textures: " << correlation << endl;
    }
    return 0;
}