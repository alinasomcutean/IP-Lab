// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}

void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
	waitKey(0);
}

//Lab 1
void changeAddGrayLevels(int value) {
	Mat img = imread("Images/flowers_24bits.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int result = img.at<uchar>(i, j) + value;
			result = result > 255 ? 255 : result;
			result = result < 0 ? 0 : result;
			img.at<uchar>(i, j) = result;
		}
	}

	imwrite("Images/newAddImage1.png", img);
	waitKey(0);
}

void changeMulGrayLevels(int value) {
	Mat img = imread("Images/flowers_24bits.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int result = img.at<uchar>(i, j) * value;
			result = result > 255 ? 255 : result;
			result = result < 0 ? 0 : result;
			img.at<uchar>(i, j) = result;
		}
	}

	imwrite("Images/newMulImage1.png", img);
	waitKey(0);
}

void createImage() {

	Mat img(256, 256, CV_8UC3);
	imshow("image", img);

	int squares[] = {
		0, 128, 0, 128, //startX, finishX, startY, finishY
		128, 256, 0, 128,
		0, 128, 128, 256,
		128, 256, 128, 256
	};

	for (int i = 0; i < 4; i++) {
		int startX = squares[i * 4 + 0];
		int finishX = squares[i * 4 + 1];
		int startY = squares[i * 4 + 2];
		int finishY = squares[i * 4 + 3];

		for (int x = startX; x < finishX; x++) {
			for (int y = startY; y < finishY; y++) {
				Vec3b pixel = img.at<Vec3b>(x, y);
				unsigned char B = pixel[0];
				unsigned char G = pixel[1];
				unsigned char R = pixel[2];

				if (i == 0) {
					B = 255;
					G = 255;
					R = 255;
				}

				if (i == 1) {
					B = 0;
					G = 0;
					R = 255;
				}

				if (i == 2) {
					B = 0;
					G = 255;
					R = 0;
				}

				if (i == 3) {
					B = 0;
					G = 255;
					R = 255;
				}

				pixel[0] = B;
				pixel[1] = G;
				pixel[2] = R;

				img.at<Vec3b>(x, y) = pixel;
			}
		}
	}

	imshow("image", img);
	waitKey(0);
}

void horizontalFlip() {
	Mat img = imread("Images/saturn.bmp", CV_LOAD_IMAGE_COLOR);
	imshow("Initial image", img);
	Vec3b aux;

	for (int i = 0; i < img.rows / 2; i++) {
		for (int j = 0; j < img.cols; j++) {
			aux = img.at<Vec3b>(i, j);
			img.at<Vec3b>(i, j) = img.at<Vec3b>(img.rows - i - 1, j);
			img.at<Vec3b>(img.rows - i - 1, j) = aux;
		}
	}

	imshow("horizontal flip", img);
	waitKey(0);
}

void verticalFlip() {
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_COLOR);
	imshow("Initial image", img);
	Vec3b aux;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols / 2; j++) {
			aux = img.at<Vec3b>(i, j);
			img.at<Vec3b>(i, j) = img.at<Vec3b>(i, img.cols - j -1);
			img.at<Vec3b>(i, img.cols - j - 1) = aux;
		}
	}

	imshow("vertical flip", img);
	waitKey(0);
}

void centerCrop() {
	Mat img = imread("Images/saturn.bmp", CV_LOAD_IMAGE_COLOR);
	Mat imgResult(img.rows / 2, img.cols / 2, CV_8UC3);

	for (int i = 0; i < imgResult.rows; i++) {
		for (int j = 0; j < imgResult.cols; j++) {
			imgResult.at<Vec3b>(i, j) = img.at<Vec3b>(i + img.rows / 4, j + img.cols / 4);
		}
	}

	imshow("Crop image", imgResult);
	imshow("Initial image", img);
	waitKey(0);
}

void imageResize(int newX, int newY) {
	Mat oldImage = imread("Images/saturn.bmp", CV_LOAD_IMAGE_COLOR);
	Mat newImage(newX, newY, CV_8UC3);

	if (newX < 0 || newY < 0) {
		newX = 1;
		newY = 1;
	}

	float ratioX = (float) oldImage.rows / newX;
	float ratioY = (float) oldImage.cols / newY;

	for (int i = 0; i < newX; i++) {
		for (int j = 0; j < newY; j++) {
			newImage.at<Vec3b>(i, j) = oldImage.at<Vec3b>(round(i * ratioX), round(j * ratioY));
		}
	}

	imshow("Initial image", oldImage);
	imshow("Resize image", newImage);
	waitKey(0);
}

//Lab 2
void rgbToMatrix() {
	Mat initImage = imread("Images/traffic_sign.png", CV_LOAD_IMAGE_COLOR);
	Mat imageB(initImage.rows, initImage.cols, CV_8UC1);
	Mat imageG(initImage.rows, initImage.cols, CV_8UC1);
	Mat imageR(initImage.rows, initImage.cols, CV_8UC1);

	for (int i = 0; i < initImage.rows; i++) {
		for (int j = 0; j < initImage.cols; j++) {
			Vec3b pixel = initImage.at<Vec3b>(i, j);

			imageB.at<uchar>(i, j) = pixel[0];
			imageG.at<uchar>(i, j) = pixel[1];
			imageR.at<uchar>(i, j) = pixel[2];
		}
	}

	imshow("Initial image", initImage);
	imshow("M1", imageB);
	imshow("M2", imageG);
	imshow("M3", imageR);
	waitKey(0);
}

void rgbToGrayscale() {
	Mat initImage = imread("Images/traffic_sign.png", CV_LOAD_IMAGE_COLOR);
	Mat resultImg(initImage.rows, initImage.cols, CV_8UC1);

	for (int i = 0; i < initImage.rows; i++) {
		for (int j = 0; j < initImage.cols; j++) {
			Vec3b pixel = initImage.at<Vec3b>(i, j);
			resultImg.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
		}
	}

	imshow("Initial image", initImage);
	imshow("Grayscale image", resultImg);
	waitKey(0);
}

void grayscaleToBlackAndWhite(int threshold) {
	Mat initImage = imread("Images/traffic_sign.png", CV_LOAD_IMAGE_COLOR);
	Mat resultImg(initImage.rows, initImage.cols, CV_8UC1);

	for (int i = 0; i < initImage.rows; i++) {
		for (int j = 0; j < initImage.cols; j++) {
			if (initImage.at<uchar>(i, j) < threshold) {
				resultImg.at<uchar>(i, j) = 0;
			}
			else {
				resultImg.at<uchar>(i, j) = 255;
			}
		}
	}

	imshow("Initial image", initImage);
	imshow("Binary image", resultImg);
	waitKey(0);
}

void hsvFromRGB() {
	Mat initImage = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);
	Mat imgH(initImage.rows, initImage.cols, CV_8UC1);
	Mat imgS(initImage.rows, initImage.cols, CV_8UC1);
	Mat imgV(initImage.rows, initImage.cols, CV_8UC1);

	for (int i = 0; i < initImage.rows; i++) {
		for (int j = 0; j < initImage.cols; j++) {
			Vec3b pixel = initImage.at<Vec3b>(i, j);
			float b = (float)pixel[0] / 255;
			float g = (float)pixel[1] / 255;
			float r = (float)pixel[2] / 255;

			float M = max(max(r, g), b);
			float m = min(min(r, g), b);
			float c = M - m;

			float value, saturation, hue;

			value = M;
			imgV.at<uchar>(i, j) = value * 255;

			if (value != 0) { //M = Value
				saturation = c / value;
			}
			else {
				saturation = 0;
			}
			imgS.at<uchar>(i, j) = saturation * 255;

			if (c != 0) {
				if (M == r) {
					hue = 60 * (g - b) / c;
				}
				if (M == g) {
					hue = 120 + 60 * (b - r) / c;
				}
				if (M == b) {
					hue = 240 + 60 * (r - g) / c;
				}
			}
			else {
				hue = 0;
			}
			if (hue < 0) {
				hue += 360;
			}
			imgH.at<uchar>(i, j) = hue * 255/360;
		}
	}

	imshow("Initial image", initImage);
	imshow("H image", imgH);
	imshow("S image", imgS);
	imshow("V image", imgV);
	waitKey(0);
}

bool isInside(Mat img, int i, int j) {
	int height = img.rows;
	int weight = img.cols;

	if (j < weight && j >= 0 && i < height && i >= 0) {
		return true;
	}
	else {
		return false;
	}
}

//Lab 3
void computeHistogram(int * h, Mat_<uchar> initImg, char* message) {
	for (int i = 0; i < initImg.rows; i++) {
		for (int j = 0; j < initImg.cols; j++) {
			int x = initImg(i, j);
			h[x]++;
		}
	}

	//showHistogram(message, h, 500, 500);
}

void computePDF(float *p, int *h, Mat_<uchar> initImg) {
	//Mat initImage = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	computeHistogram(h, initImg, "histogram");

	for (int i = 0; i < 256; i++) {
		p[i] = (float)*(h + i) / (initImg.rows * initImg.cols);
	}
}

void multiLevelThresholding(float *p, int *h, int wh, float th) {
	Mat initImg = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat resultImg(initImg.rows, initImg.cols, CV_8UC1);

	//Step 1
	int length = 0;
	int *maxim = (int*)malloc(256 * sizeof(int));
	maxim[0] = 0;

	computePDF(p, h, initImg);

	for (int k = 0 + wh; k <= 255 - wh; k++) {
		float sum = 0;
		bool greater = false;

		for (int j = k - wh; j <= k + wh; j++) {
			sum += p[j];
			if (p[k] >= p[j]) {
				greater = true;
			}
		}

		float v = sum / (2 * wh + 1);

		if ((p[k] > (v + th)) && greater) {
			length++;
			maxim[length] = k;
		}
	}

	length++;
	maxim[length] = 255;

	//Step 2

	for (int i = 0; i < initImg.rows; i++) {
		for (int j = 0; j < initImg.cols; j++) {
			int pixel = initImg.at<uchar>(i, j);
			int minim = 255;
			int position = length;

			for (int k = 0; k <= length; k++) {
				if (abs(pixel - maxim[k]) < minim) {
					minim = abs(pixel - maxim[k]);
					position = k;
				}
			}

			resultImg.at<uchar>(i, j) = maxim[position];
		}
	}

	imshow("Init image", initImg);
	imshow("Result image", resultImg);
	waitKey(0);
}

void floydSteinberg(float *p, int *h, int wh, float th) {
	Mat initImg = imread("Images/saturn.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat resultImg(initImg.rows, initImg.cols, CV_8UC1);

	//Step 1
	int length = 0;
	int *maxim = (int*)malloc(256 * sizeof(int));
	maxim[0] = 0;

	computePDF(p, h, initImg);

	for (int k = 0 + wh; k <= 255 - wh; k++) {
		float sum = 0.0f;
		bool greater = false;

		for (int j = k - wh; j <= k + wh; j++) {
			sum += p[j];
			if (p[k] >= p[j]) {
				greater = true;
			}
		}

		float v = sum / (2 * wh + 1);

		if ((p[k] > (v + th)) && greater) {
			length++;
			maxim[length] = k;
		}
	}

	length++;
	maxim[length] = 255;

	//Step 2

	for (int i = 0; i < initImg.rows; i++) {
		for (int j = 0; j < initImg.cols; j++) {
			resultImg.at<uchar>(i, j) = initImg.at<uchar>(i, j);
		}
	}

	for (int i = 0; i < resultImg.rows; i++) {
		for (int j = 0; j < resultImg.cols; j++) {
			int oldPixel = resultImg.at<uchar>(i, j);
			int minim = 255;
			int position = 0;

			for (int k = 0; k <= length; k++) {
				if (abs(oldPixel - maxim[k]) < minim) {
					minim = abs(oldPixel - maxim[k]);
					position = k;
				}
			}

			int newPixel = maxim[position];
			resultImg.at<uchar>(i, j) = newPixel;
			int error = oldPixel - newPixel;

			if (isInside(resultImg, i, j + 1)) {
				resultImg.at<uchar>(i, j + 1) = max(0, min(255, resultImg.at<uchar>(i, j+1) + 7 * error / 16));
			}
			if (isInside(resultImg, i + 1, j - 1)) {
				resultImg.at<uchar>(i + 1, j - 1) = max(0, min(255, resultImg.at<uchar>(i+1, j-1) + 3 * error / 16));
			}
			if (isInside(resultImg, i + 1, j)) {
				resultImg.at<uchar>(i + 1, j) = max(0, min(255, resultImg.at<uchar>(i+1, j) + 5 * error / 16));
			}
			if (isInside(resultImg, i + 1, j + 1)) {
				resultImg.at<uchar>(i + 1, j + 1) = max(0, min(255, resultImg.at<uchar>(i+1, j+1) + error / 16));
			}
		}
	}

	imshow("Init image", initImg);
	imshow("Result image", resultImg);
	waitKey(0);
}

//Lab 4
int computeArea(Mat img, int x, int y) {
	
	int area = 0;
	Vec3b color = img.at<Vec3b>(y,x);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i,j);
			if (color == pixel) {
				area++;
			}
		}
	}
	return area;
}

void computeCenterOfMass(Mat img, int *pos, int x, int y, int area) {

	Vec3b color = img.at<Vec3b>(y, x);
	int rowAux = 0, colAux = 0;
	int row, col;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			if (color == pixel) {
				rowAux += i;
				colAux += j;
			}
		}
	}

	row = rowAux / area;
	col = colAux / area;
	pos[0] = row;
	pos[1] = col;
}

double computeAxisOfElongation(Mat img, int row, int col, int x, int y, Mat resultImg) {

	int nominator = 0, denominator1 = 0, denominator2 = 0;
	int cMax = 0, cMin = img.cols;

	Vec3b color = img.at<Vec3b>(y, x);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			if (color == pixel) {
				nominator += (i - row)*(j - col)*img.at<uchar>(i,j);
				denominator1 += pow((j - col), 2)*img.at<uchar>(i, j);
				denominator2 += pow((i - row), 2)*img.at<uchar>(i, j);

				//draw elongation line
				if (j > cMax) {
					cMax = j;
				}
				if (j < cMin) {
					cMin = j;
				}
			}
		}
	}

	float nom = 2 * nominator;
	float denominator = (denominator1 - denominator2);
	double phi = atan2(nom, denominator) / 2;

	double elongation = phi * 180 / PI;
	if (phi < 0) {
		elongation += 180;
	}

	float tang = tan(phi * CV_PI / 180);
	Point A(cMin, tang * (cMin - col) + row);
	Point B(cMax, tang * (cMax - col) + row);
	line(resultImg, A, B, color);

	return elongation;
}

int computePerimeterAndDrawTheContour(Mat img, int x, int y, Mat resultImg) {

	Vec3b white;
	white[0] = 255;
	white[1] = 255;
	white[2] = 255;

	int perimeter = 0;

	Vec3b color = img.at<Vec3b>(y, x);

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			if (color == pixel) {
				if (img.at<Vec3b>(i - 1, j - 1) == white || img.at<Vec3b>(i - 1, j) == white || img.at<Vec3b>(i - 1, j + 1) == white ||
				img.at<Vec3b>(i, j - 1) == white || img.at<Vec3b>(i, j + 1) == white || img.at<Vec3b>(i + 1, j - 1) == white ||
				img.at<Vec3b>(i + 1, j) == white || img.at<Vec3b>(i + 1, j + 1) == white)
				{
					perimeter++;
					resultImg.at<Vec3b>(i, j) = img.at<Vec3b>(i, j); 
				}
			}
		}
	}

	return perimeter * PI / 4;
}

double computeThinessRatio(Mat img, int area, int perimeter) {
	return 4 * PI * (area / pow(perimeter, 2));
}

double computeAspectRatio(Mat img, int x, int y) {

	int rMin = img.rows, rMax = 0; 
	int cMin = img.cols, cMax = 0;

	Vec3b color = img.at<Vec3b>(y, x);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			if (color == pixel) {
				if (rMin > i) {
					rMin = i;
				}
				if (rMax < i) {
					rMax = i;
				}
				if (cMin > j) {
					cMin = j;
				}
				if (cMax < j) {
					cMax = j;
				}
			}
		}
	}

	return (float)(cMax - cMin + 1) / (rMax - rMin + 1);
}

void computeProjections(Mat img, int x, int y, Mat resultImg) {

	Vec3b color = img.at<Vec3b>(y, x);

	int* h = (int*)malloc(img.rows * (sizeof(int)));
	int* v = (int*)malloc(img.cols * (sizeof(int)));

	for (int i = 0; i < img.rows; i++) {
		v[i] = 0;
	}

	for (int i = 0; i < img.cols; i++) {
		h[i] = 0;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			if (color == pixel) {
				h[j]++;
				v[i]++;
			}
		}
	}

	for (int i = 0; i < img.rows; i++) {
		cv::line(resultImg, Point(0, i), Point(v[i], i), color, 1);
	}

	for (int i = 0; i < img.cols; i++) {
		cv::line(resultImg, Point(i, 0), Point(i, h[i]), color, 1);
	}
}

void computeParamMultipleObjects(int event, int x, int y, int flags, void* param) {
	
	Mat* initImg = (Mat*)param;
	int *pos = (int*)malloc(2 * sizeof(int));
	int row = 0, col = 0;

	Mat resultImg((*initImg).rows, (*initImg).cols, CV_8UC3);
	Mat projections((*initImg).rows, (*initImg).cols, CV_8UC3);

	if (event == EVENT_LBUTTONDBLCLK) {
		int area = computeArea(*initImg, x, y);
		printf("\nArea: %d\n", area);

		computeCenterOfMass(*initImg, pos, x, y, area);
		cv::circle(resultImg, Point(pos[1], pos[0]), 2, Scalar(0, 0, 0));
		//line(resultImg, Point(pos[0], pos[1]), Point(10,10), 0, 1, 0);
		printf("Center of mass: row %d, column %d\n", pos[0], pos[1]);

		double elongation = computeAxisOfElongation(*initImg, pos[0], pos[1], x, y, resultImg);
		printf("Axis of elongation: %f\n", elongation);

		int perimeter = computePerimeterAndDrawTheContour(*initImg, x, y, resultImg);
		printf("Perimeter: %d\n", perimeter);

		double thinessRatio = computeThinessRatio(*initImg, area, perimeter);
		printf("Thinnes ratio: %f\n", thinessRatio);

		double aspectRatio = computeAspectRatio(*initImg, x, y);
		printf("Aspect ratio: %f\n", aspectRatio);

		computeProjections(*initImg, x, y, projections);

		imshow("Contour, elongation line, center of mass, projections", resultImg);
		imshow("Projections", projections);
		waitKey(0);
	}
}

void computationForObject(String windowName) {
	Mat initImg;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		initImg = imread(fname);
		namedWindow(windowName, 1);
		setMouseCallback(windowName, computeParamMultipleObjects, &initImg);
		imshow(windowName, initImg);
		waitKey(0);
	}
}

//Lab 5
void displayLabels(Mat_<uchar> labels) {
	int height = labels.rows;
	int width = labels.cols;
	Mat_<Vec3b> resultImg(height, width);

	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);

	int r[256], g[256], b[256];

	//always white
	r[0] = 255;
	g[0] = 255;
	b[0] = 255;

	for (int i = 1; i < 256; i++) {
		r[i] = d(gen);
		g[i] = d(gen);
		b[i] = d(gen);
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			resultImg.at<Vec3b>(i, j)[0] = b[labels(i, j)];
			resultImg.at<Vec3b>(i, j)[1] = g[labels(i, j)];
			resultImg.at<Vec3b>(i, j)[2] = r[labels(i, j)];
		}
	}

	imshow("Label image", resultImg);
}

void bfs(int n) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		Mat_<uchar> initImage = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImage.rows;
		int width = initImage.cols;

		uchar label = 0;
		Mat_<uchar> labels = Mat::zeros(height, width, CV_8UC1);
		int dx[]{ 0, 0, -1, 1, 1, 1, -1, -1 };
		int dy[]{ -1, 1, 0, 0, 1, -1, 1, -1 };

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (initImage(i, j) == 0 && labels(i,j) == 0) {
					label++;
					std::queue<Point2i> Q;
					labels(i,j) = label;
					Q.push({j, i});

					while (!Q.empty()) {
						Point2i p = Q.front();
						Q.pop();

						for (int k = 0; k < n; k++) {
							int x = p.x + dx[k];
							int y = p.y + dy[k];
							if (isInside(initImage, y, x) && labels(y,x) == 0 && initImage(y,x) == 0) {
								labels(y, x) = label;
								Q.push({x, y});
							}
						}
					}
				}
			}
		}
		displayLabels(labels);
		waitKey(0);
	}
}

void twoPassLabeling() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		Mat_<uchar> initImage = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImage.rows;
		int width = initImage.cols;

		Mat_<Vec3b> partialResultImg(height, width);
		Mat_<Vec3b> resultImg(height, width);

		uchar label = 0;
		Mat_<uchar> labels = Mat::zeros(height, width, (uchar)0);

		std::vector<std::vector<int>> edges;
		edges.resize(5000);

		int dx[]{ 0, 0, -1, 1, 1, 1, -1, -1 };
		int dy[]{ -1, 1, 0, 0, 1, -1, 1, -1 };

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (initImage(i, j) == 0 && labels(i, j) == 0) {
					std::vector<int> L;
					for (int k = 0; k < 8; k++) {
						int x = j + dx[k];
						int y = i + dy[k];
						if (isInside(initImage, y, x) && labels(y, x) > 0) {
							L.push_back(labels(y, x));
						}
					}

					if (L.size() == 0) { //assign new label
						label++;
						labels(i, j) = label;
					} 
					else { //assign smallest neighbor
						int minim = INT_MAX;
						for (int k = 0; k < L.size(); k++) {
							if (L[k] < minim) {
								minim = L[k];
							}
						}

						labels(i, j) = minim;
						for (int y = 0; y < L.size(); y++) {
							if (minim != L[y]) {
								edges[minim].push_back(L[y]);
								edges[L[y]].push_back(minim);
							}
						}
					}
				}
			}
		}

		displayLabels(labels);
		
		uchar newLabel = 0;
		int *newLabels = new int[label + 1];

		for (int i = 0; i < label+1; i++)
		{
			newLabels[i] = 0;
		}

		for (int i = 1; i <= label; i++) {
			if (newLabels[i] == 0) {
				newLabel++;
				std::queue<int> Q;
				newLabels[i] = newLabel;
				Q.push(i);

				while (!Q.empty()) {
					int x = Q.front();
					Q.pop();

					for (int y = 0; y < edges[x].size(); y++) {
						if (newLabels[y] == 0) {
							newLabels[y] = newLabel;
							Q.push(y);
						}
					}
				}
			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				labels(i, j) = newLabels[labels(i, j)];
			}
		}

		imshow("Initial image", initImage);
		waitKey(0);
	}
}

//Lab 6
void borderTracing(){
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;


		Mat_<uchar> resultImg(height, width);

		int dir = 7; 
		int dx[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int dy[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		bool startPoint = true;

		Point2d p0; // starting point

		for (int i = 0; i < height && startPoint; i++) {
			for (int j = 0; j < width && startPoint; j++) {
				if (initImg(i, j) == 0) {
					startPoint = false;
					p0.x = j;
					p0.y = i;
				}
			}
		}
		std::cout << p0.x << " " << p0.y;

		std::vector<Point2d> border;
		std::vector<int> derivatives;
		std::vector<int> chain;

		border.push_back(p0);

		while (!(border.size() > 2 && border[border.size() - 1] == border[1] && border[border.size() - 2] == border[0])) {
		//while (border.size() <= 2 || border[0] != border[border.size() - 2] || border[1] != border[border.size() - 1]) {
			int newDir;

			if (dir % 2 == 0) {
				newDir = (dir + 7) % 8;
			}
			else {
				newDir = (dir + 6) % 8;
			}

			//Find coordinates for the current pixel
			int currentX = border[border.size() - 1].x;
			int currentY = border[border.size() - 1].y;

			for (int i = 0; i < 8; i++) {
				//neghbors coordinates
				int newX = currentX + dx[newDir];
				int newY = currentY + dy[newDir];

				//Pixel is inside the image
				if (isInside(initImg, newY, newX)) {
					//Current pixel is black
					if (initImg(newY, newX) == 0) {
						//Pixel is a valid one
						//Add it to the border vector
						border.push_back(Point2d(newX, newY));
						chain.push_back(newDir);
						derivatives.push_back((newDir - dir + 8) % 8);
						dir = newDir;
						break;
					}
				}

				newDir = (newDir + 1) % 8;
			}

		}

		//delete last 2 element from the border because they are the same with the first 2
		border.pop_back();
		border.pop_back();

		//delete last 2 element from the chain because they are the same with the first 2
		chain.pop_back();
		chain.pop_back();

		//display the border
		for (int i = 0; i < border.size(); i++) {
			resultImg(border[i].y, border[i].x) = 0;
		}

		//display the chain code
		std::cout << "Chain code: \n";
		for (int i = 0; i < chain.size(); i++) {
			printf("%d ", chain[i]);
		}

		std::cout << "\n\nDerivative code: \n";
		for (int i = 0; i < derivatives.size(); i++) {
			printf("%d ", derivatives[i]);
		}

		imshow("Init image", initImg);
		imshow("Border", resultImg);
		waitKey(0);
	}
}

void buildBorder() {
	Mat_<uchar> initImage = imread("Images/Border_Tracing/gray_background.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	FILE* file = fopen("reconstruct.txt", "r");
	
	if (file != NULL) {
		Point p0;
		int chainCodes;
		int dx[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		int dy[] = { 0, -1, -1, -1, 0, 1, 1, 1 };

		fscanf(file, "%d", &p0.x);
		fscanf(file, "%d", &p0.y);
		fscanf(file, "%d", &chainCodes);

		initImage(p0.y, p0.x) = 0;

		Point current = p0;
		for (int i = 0; i < chainCodes; i++) {
			int dir;
			fscanf(file, "%d", &dir);

			Point newPixel;
			newPixel.x = current.x + dx[dir];
			newPixel.y = current.y + dy[dir];
			
			initImage(newPixel.y, newPixel.x) = 0;

			current = newPixel;
		}

		imshow("Border from chain code", initImage);
		waitKey(0);
	}
	else {
		printf("File cannot be open! \n\n");
	}
}

//Lab 7
Mat_<uchar> dilation(Mat_<uchar> initImg) {

	int strElem[3][3] = { {1,1,1},{1,0,1}, {1,1,1} };	
	int height = initImg.rows;
	int width = initImg.cols;
	Mat_<uchar> resultImg = initImg.clone();

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (initImg(i, j) == 0) { // if the current pixel is black 
				for (int k1 = 0; k1 < 3; k1++) {
					for (int k2 = 0; k2 < 3; k2++) {
						if (strElem[k1][k2] == 1) { // for all the 8 neighboors 
							resultImg(i - 1 + k2, j - 1 + k1) = 0; // color them
						}
					}
				}
			}
		}
	}

	return resultImg;
}

Mat_<uchar> erosion(Mat_<uchar> initImg) {

	int strElem[3][3] = { {1,1,1},{1,0,1}, {1,1,1} };
	int height = initImg.rows;
	int width = initImg.cols;
	Mat_<uchar> resultImg = initImg.clone();

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (initImg(i, j) == 0) { // if the current pixel is white 
				for (int k1 = 0; k1 < 3; k1++) {
					for (int k2 = 0; k2 < 3; k2++) {
						if (strElem[k1][k2] == 1) { // for all the 8 neighboors 
							if (initImg(i - 1 + k2, j - 1 + k1) == 255) { // i have at least 1 white neighboor
								resultImg(i, j) = 255; // color the origin pixel in white
							}
						}
					}
				}
			}
		}
	}

	return resultImg;
}

Mat_<uchar> opening(Mat_<uchar> initImg) {
	Mat_<uchar> resultImg = erosion(initImg); // apply erosion
	resultImg = dilation(resultImg); // apply dilation
	return resultImg;
}

Mat_<uchar> openingTimes(Mat_<uchar> initImg, int n) {
	Mat_<uchar> result;
	Mat_<uchar> aux;
	result = opening(initImg); // find the open image one time
	aux = result;

	for (int i = 1; i < n; i++) { // for more than 1, apply it repeteadly
		result = opening(result);
		aux = result;
	}

	return result;
}

Mat_<uchar> closing(Mat_<uchar> initImg) {
	Mat_<uchar> resultImg = dilation(initImg); // apply dilation
	resultImg = erosion(resultImg); // apply erosion
	return resultImg;
}

Mat_<uchar> closingTimes(Mat_<uchar> initImg, int n) {
	Mat_<uchar> result;
	Mat_<uchar> aux;
	result = closing(initImg); // find the close image one time
	aux = result;

	for (int i = 1; i < n; i++) { // for more than 1, apply it repeteadly
		result = closing(result);
		aux = result;
	}

	return result;
}

Mat_<uchar> difference(Mat_<uchar> x, Mat_<uchar> y) { // x-initial image, y-imagea affter erosion
	int height = x.rows;
	int width = y.cols;

	Mat_<uchar> result(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (y(i, j) == 255) { // if the pixel from erosion image is white
				result(i, j) = x(i, j); // in the final image, the pixel has the same color as initial image
			}
			else { // if the pixel from erosion is black
				if (x(i, j) == 0) { // if the pixel from initial image is black, means that we are inside the object
					result(i, j) = 255; // color the pixel white
				}
			}
		}
	}

	return result;
}

void boundaryExtraction() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;
		Mat_<uchar> resultImg(height, width);

		resultImg = erosion(initImg); // apply erosion on initial image
		resultImg = difference(initImg, resultImg); // find the border

		imshow("Initial image", initImg);
		imshow("Boundary extraction", resultImg);
		waitKey(0);
	}
}

Mat_<uchar> complementary(Mat_<uchar> image) {
	int height = image.rows;
	int width = image.cols;

	Mat_<uchar> resultImg(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (image(i, j) == 0) { // if pixel is black make it white in the result
				resultImg(i, j) = 255;
			}
			else { // otherwise make it black in the result
				resultImg(i, j) = 0;
			}
		}
	}

	return resultImg;
}

Mat_<uchar> intersection(Mat_<uchar> x, Mat_<uchar> y)
{
	int height = x.rows;
	int width = x.cols;
	Mat_<uchar> result(height, width);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (x(i, j) == 0 && y(i, j) == 0) // if both pixels are black, make it black
				result(i, j) = 0
;
			else // otherwise, make it white
				result(i, j) = 255;
		}
	}
	return result;
}

Mat_<uchar> unionOperation(Mat_<uchar> x, Mat_<uchar> y)
{
	int height = x.rows;
	int width = x.cols;
	Mat_<uchar> result(height, width);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (x(i, j) == 0 || y(i, j) == 0) // if at least one pixel is black, make it black
				result(i, j) = 0;
			else // otherwise, make it white
				result(i, j) = 255;
		}
	}
	return result;
}

bool equals(Mat_<uchar> x, Mat_<uchar> y)
{
	int height = x.rows;
	int width = x.cols;
	Mat_<uchar> result(height, width);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (x(i, j) != y(i, j))
				return false;
		}
	}
	return true;
}

void filling() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE); // initial image
		int height = initImg.rows;
		int width = initImg.cols;

		Mat_<uchar> resultImg(height, width); // the final image
		Mat_<uchar> complementImg(height, width); // the complementary image
		Mat_<uchar> partial(height, width); // a partial image
		bool eq = false;

		//create a white output image
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				resultImg(i, j) = 255;
			}
		}

		// make the middle pixel black
		resultImg(height / 2, width / 2) = 0;

		//complement the source image
		complementImg = complementary(initImg);

		do {
			partial = resultImg.clone();
			resultImg = dilation(resultImg);   
			resultImg = intersection(resultImg, complementImg); 
			eq = equals(partial, resultImg);
		} while (eq == false);

		resultImg = unionOperation(initImg, resultImg);

		imshow("Initial image", initImg);
		imshow("Filling image", resultImg);
		waitKey(0);
	}
}

void showImages(Mat_<uchar> initImg, Mat_<uchar> resultImg, String title) {
	imshow("Initial image", initImg);
	imshow(title, resultImg);
	waitKey(0);
}

// Lab 8
float computeMeanValue(Mat_<uchar> initImg) {
	int height = initImg.rows;
	int width = initImg.cols;
	float meanValue = 0;
	int M = height * width;
	int sum = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			sum += initImg(i, j);
		}
	}

	meanValue = sum / M;
	printf("Mean value is %f\n", meanValue);
	return meanValue;
}

float computeStandardDeviation() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;

		float meanValue = computeMeanValue(initImg);
		int M = height * width;
		int sum = 0;
		float standardDeviation = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				sum += pow(initImg(i, j) - meanValue, 2);
			}
		}

		standardDeviation = sqrt(sum / M);
		printf("Standard deviation is %f\n", standardDeviation);
		return standardDeviation;
	}
}

void thresholdingAlgorithm() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;
		Mat_<uchar> resultImg = initImg.clone();

		int h[256] = { 0 };
		int T, n1 = 0, n2 = 0;
		int sum1 = 0, sum2 = 0;
		int g1 = 0, g2 = 0;

		// Compute the image histogram
		computeHistogram(h, initImg, "Histogram");

		// Get initial maxim and minim
		int maxim = initImg(0,0), minim = initImg(0, 0);

		// Find the maximum and minimum intensity in the image
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (initImg(i, j) > maxim) {
					maxim = initImg(i, j);
				}
				if (initImg(i, j) < minim) {
					minim = initImg(i, j);
				}
			}
		}

		// Initial value for thresholding
		T = (maxim + minim) / 2;
		int newT = T;

		// Segment the image after T by dividing the image pixels into 2 groups
		do {
			// Keep the previous value for thresholding
			T = newT;

			// Compute the means using the initial histogram
			for (int i = minim; i <= T; i++) {
				n1 += h[i];
				sum1 += i * h[i];
			}
			for (int i = T + 1; i <= maxim; i++) {
				n2 += h[i];
				sum2 += i * h[i];
			}

			g1 = sum1 / (float)n1;
			g2 = sum2 / (float)n2;

			// Update the threshold value
			newT = (g1 + g2) / 2;
		} while (abs(newT - T) >= 0.1);

		// Threshold the image using T
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (initImg(i, j) <= T) {
					resultImg(i, j) = 0;
				}
				if (initImg(i, j) > T) {
					resultImg(i, j) = 255;
				}
			}
		}

		printf("Thresholding is %d\n", newT);
		showImages(initImg, resultImg, "Thresholding algorithm");
	}
}

void histogramSlide(int offset) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;
		Mat_<uchar> resultImg = initImg.clone();

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int out = initImg(i, j) + offset;
				if (out <= 255 && out >= 0) {
					resultImg(i, j) = out;
				}
				else {
					if (out > 255) {
						resultImg(i, j) = out - 255;
					}
					else {
						resultImg(i, j) = 255 + out;
					}
				}
			}
		}

		showImages(initImg, resultImg, "Histogram slide");
	}
}

void stretchingShrinking(int gOutMin, int gOutMax) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;
		Mat_<uchar> resultImg(height, width);

		//Find the minimum and maximum from the image
		int gInMax = initImg(0, 0), gInMin = initImg(0, 0);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (initImg(i, j) > gInMax) {
					gInMax = initImg(i, j);
				}
				if (initImg(i, j) < gInMin) {
					gInMin = initImg(i, j);
				}
			}
		}

		float raport;
		raport = (float) (gOutMax - gOutMin) / (gInMax - gInMin);

		if (raport > 1) {
			printf("Stretch\n");
		}
		if (raport < 1) {
			printf("Shrink\n");
		}

		//Compute the final image
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				resultImg(i, j) = gOutMin + (initImg(i, j) - gInMin) * raport;
			}
		}

		//Show the histograms for both images
		int h1[256] = { 0 };
		computeHistogram(h1, initImg, "Initial image histogram");
		int h2[256] = { 0 };
		computeHistogram(h2, resultImg, "Result image histogram");
		showImages(initImg, resultImg, "Stretch and Shrink");
	}
}

void gammaCorrection(float gamma) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;
		Mat_<uchar> resultImg(height, width);
		int L = 255;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float out = L * pow( initImg(i, j) / (float) L, gamma);
				if (out >= 0 && out <= 255) {
					resultImg(i, j) = out;
				}
				else {
					if (out > 255) {
						resultImg(i, j) = out - 255;
					}
					else {
						resultImg(i, j) = 255 + out;
					}
				}
			}
		}
		
		showImages(initImg, resultImg, "Gamma correction");
	}
}

void histogramEqualization() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;
		Mat_<uchar> resultImg(height, width);

		int M = height * width;
		//Show the histogram for the initial image
		int h1[256] = { 0 };
		computeHistogram(h1, initImg, "Histogram of the initial image");

		//Find the cumulative histogram
		int hCumulative[256] = { 0 };
		int sum = 0;
		for (int i = 0; i < 256; i++) {
			sum += h1[i];
			hCumulative[i] = sum;
		}

		//Compute the final image
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				resultImg(i, j) = (float)255 / M * hCumulative[initImg(i, j)];
			}
		}

		//Show the histogram for the resulting image
		int h2[256] = { 0 };
		computeHistogram(h2, resultImg, "Histogram of the result image");

		showImages(initImg, resultImg, "Histogram equalization");
	}

}

// Lab 9
void convolution(Mat_<int> &filter, Mat_<uchar> &img, Mat_<uchar> &output) {

	int height = img.rows;
	int width = img.cols;

	output.create(img.size());
	memcpy(output.data, img.data, height * width * sizeof(uchar));

	int scalingCoeff = 1;
	int additionFactor = 0;
	int w = filter.rows;
	int k = (w - 1) / 2;

	// decide if the filter is low pass or high pass and compute the scaling coefficient and the addition factor
	// low pass if all elements >= 0
	// high pass has elements < 0
	int sPlus = 0, sMinus = 0;
	bool lowPass = true;
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			if (filter(i, j) >= 0) {
				sPlus += filter(i, j);
			}
			else {
				lowPass = false;
				sMinus += filter(i, j);
			}
		}
	}

	// compute scaling coefficient and addition factor for low pass and high pass
	// low pass: additionFactor = 0, scalingCoeff = sum of all elements
	// high pass: formula 9.20

	// implement convolution operation (formula 9.2)
	// do not forget to divide with the scaling factor and add the addition factor in order to have values between [0, 255]

	if (lowPass) { // Low pass filter
		additionFactor = 0;
		scalingCoeff = sPlus;
		for (int i = k; i < height - k; i++) {
			for (int j = k; j < width - k; j++) {
				int sum = 0;
				for (int u = 0; u < w; u++) {
					for (int v = 0; v < w; v++) {
						sum += filter(u, v) * img(i + u - k, j + v - k);
					}
				}
				output(i, j) = sum / scalingCoeff + additionFactor;
			}
		}
	}
	else { // High pass filter
		scalingCoeff = 2 * max(sPlus, abs(sMinus));
		additionFactor = 255 / 2;
		for (int i = k; i < height - k; i++) {
			for (int j = k; j < width - k; j++) {
				int F = 0;
				for (int u = 0; u < w; u++) {
					for (int v = 0; v < w; v++) {
						F += filter(u, v) * img(i + u - 1, j + v - 1);
					}
				}
				output(i, j) = F / scalingCoeff + additionFactor;
			}
		}
	}
}

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat addLog(Mat src) {
	Mat result = src;
	result += Scalar::all(1);
	log(result, result);
	normalize(result, result, 0, 1, NORM_MINMAX);
	return result;
}

Mat idealLowPassFilter(Mat channel, int R) {
	Mat result = Mat(channel.size(), CV_32F);
	for (int i = 0; i < channel.rows; i++) {
		for (int j = 0; j < channel.cols; j++) {
			float x = pow(channel.rows / 2 - i, 2) + pow(channel.cols / 2 - j, 2);
			if (x <= pow(R,2)) {
				result.at<float>(i, j) = channel.at<float>(i, j);
			}
			else {
				result.at<float>(i, j) = 0;
			}
		}
	}
	return result;
}

Mat idealHighPassFilter(Mat channel, int R) {
	Mat result = Mat(channel.size(), CV_32F);
	for (int i = 0; i < channel.rows; i++) {
		for (int j = 0; j < channel.cols; j++) {
			float x = pow(channel.rows / 2 - i, 2) + pow(channel.cols / 2 - j, 2);
			if (x > pow(R, 2)) {
				result.at<float>(i, j) = channel.at<float>(i, j);
			}
			else {
				result.at<float>(i, j) = 0;
			}
		}
	}
	return result;
}

Mat gaussianLowPassFilter(Mat channel, float A) {
	Mat result = Mat(channel.size(), CV_32F);
	for (int i = 0; i < channel.rows; i++) {
		for (int j = 0; j < channel.cols; j++) {
			float exponent = (pow(channel.rows / 2 - i, 2) + pow(channel.cols / 2 - j, 2)) / (pow(A,2));
				result.at<float>(i, j) = channel.at<float>(i, j) * exp(-exponent);
		}
	}
	return result;
}

Mat gaussianHighPassFilter(Mat channel, float A) {
	Mat result = Mat(channel.size(), CV_32F);
	for (int i = 0; i < channel.rows; i++) {
		for (int j = 0; j < channel.cols; j++) {
			float exponent = (pow(channel.rows / 2 - i, 2) + pow(channel.cols / 2 - j, 2)) / (pow(A, 2));
			result.at<float>(i, j) = channel.at<float>(i, j) * (1 - exp(-exponent));
		}
	}
	return result;
}

Mat generic_frequency_domain_filter(char* option, Mat src) {
	// Discrete Fourier Transform: https://docs.opencv.org/4.2.0/d8/d01/tutorial_discrete_fourier_transform.html
	int height = src.rows;
	int width = src.cols;

	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	// Centering transformation
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	// the frequency is represented by its real and imaginary parts called frequency coefficients
	// split into real and imaginary channels fourier(i, j) = Re(i, j) + i * Im(i, j)
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);  // channels[0] = Re (real part), channels[1] = Im (imaginary part)

	//calculate magnitude and phase of the frequency by transforming it from cartesian to polar coordinates
	// the magnitude is useful for visualization

	Mat mag, phi;
	magnitude(channels[0], channels[1], mag); // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6d3b097586bca4409873d64a90fe64c3
	phase(channels[0], channels[1], phi); // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9db9ca9b4d81c3bde5677b8f64dc0137

	// Display here the log of magnitude (Add 1 to the magnitude to avoid log(0)) (see image 9.4e))
	// do not forget to normalize
	Mat resultMag = addLog(mag);
	imshow("Centered logarithm of magnitude spectra", mag);
	waitKey(0);

	// Insert filtering operations here ( channels[0] = Re(DFT(I), channels[1] = Im(DFT(I) )
	if (strcmp(option, "low_pass") == 0) { // Ideal low pass filter
		channels[0] = idealLowPassFilter(channels[0], 20);
		channels[1] = idealLowPassFilter(channels[1], 20);
	}
	else {
		if (strcmp(option, "high_pass") == 0) {
			channels[0] = idealHighPassFilter(channels[0], 20);
			channels[1] = idealHighPassFilter(channels[1], 20);
		}
		else {
			if (strcmp(option, "gaussian_low_pass") == 0) {
				channels[0] = gaussianLowPassFilter(channels[0], 20);
				channels[1] = gaussianLowPassFilter(channels[1], 20);
			}
			else {
				if (strcmp(option, "gaussian_high_pass") == 0) {
					channels[0] = gaussianHighPassFilter(channels[0], 20);
					channels[1] = gaussianHighPassFilter(channels[1], 20);
				}
			}
		}
	}

	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT);

	// Inverse Centering transformation
	centering_transform(dstf);

	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	return dst;
}

//Lab 10
void createFilterVector(Mat_<uchar> initImg, Mat_<uchar> resultImg, int positionX, int positionY, int filterSize) {
	std::vector<uchar> vector;

	// Compute the filter vector
	for (int i = 0; i < filterSize; i++) {
		for (int j = 0; j < filterSize; j++) {
			int posX = positionX + i - filterSize / 2;
			int posY = positionY + j - filterSize / 2;
			if (isInside(initImg, posX, posY)) {
				vector.push_back(initImg(posX, posY));
			}
		}
	}

	// Sort the vector
	std::sort(vector.begin(), vector.end());

	// Find the middle value from the sorted vector
	resultImg(positionX, positionY) = vector[vector.size() / 2];
}

void medianFilter(int filterSize) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;
		Mat_<uchar> resultImg(height, width);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				// Compute filter for each pixel
				createFilterVector(initImg, resultImg, i, j, filterSize);
			}
		}

		imshow("Initial image", initImg);
		imshow("Result image", resultImg);
		waitKey(0);
	}
}

Mat_<float> computeG(int filterSize) {
	Mat_<float> G(filterSize, filterSize);
	for (int i = 0; i < filterSize; i++) {
		for (int j = 0; j < filterSize; j++) {
			// Compute the formula for Gaussian filter
			float exponent = pow(i - filterSize / 2.0, 2) + pow(j - filterSize / 2.0, 2);
			float g = exp(-exponent / (2.0 * pow(filterSize / 6.0, 2))) / (2.0 * PI * pow(filterSize / 6.0, 2));
			G(i,j) = g;
		}
	}

	return G;
}

void convolutionFloat(Mat_<float> &filter, Mat_<uchar> &img, Mat_<uchar> &output) {

	int height = img.rows;
	int width = img.cols;

	output.create(img.size());
	memcpy(output.data, img.data, height * width * sizeof(uchar));

	float scalingCoeff = 1;
	float additionFactor = 0;
	int w = filter.rows;
	int k = (w - 1) / 2;

	float sPlus = 0, sMinus = 0;
	bool lowPass = true;
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			if (filter(i, j) >= 0) {
				sPlus += filter(i, j);
			}
			else {
				lowPass = false;
				sMinus += filter(i, j);
			}
		}
	}

	if (lowPass) { // Low pass filter
		additionFactor = 0;
		scalingCoeff = sPlus;
		for (int i = k; i < height - k; i++) {
			for (int j = k; j < width - k; j++) {
				float sum = 0;
				for (int u = 0; u < w; u++) {
					for (int v = 0; v < w; v++) {
						sum += filter(u, v) * img(i + u - k, j + v - k);
					}
				}
				output(i, j) = sum / scalingCoeff + additionFactor;
			}
		}
	}
	else { // High pass filter
		scalingCoeff = 2 * max(sPlus, abs(sMinus));
		additionFactor = 255 / 2;
		for (int i = k; i < height - k; i++) {
			for (int j = k; j < width - k; j++) {
				float F = 0;
				for (int u = 0; u < w; u++) {
					for (int v = 0; v < w; v++) {
						F += filter(u, v) * img(i + u - 1, j + v - 1);
					}
				}
				output(i, j) = F / scalingCoeff + additionFactor;
			}
		}
	}
}

void gaussianFilter2D(Mat_<uchar> initImg, int filterSize, Mat_<uchar> resultImg) {
	//char fname[MAX_PATH];
	//while (openFileDlg(fname)) {
		//Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;
		resultImg(height, width);

		Mat_<float> G = computeG(filterSize);
		convolutionFloat(G, initImg, resultImg);

		imshow("Initial image", initImg);
		imshow("Result image", resultImg);
		waitKey(0);
	//}
}

void gaussianFilterUsingGaussianKernel(int filterSize) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;
		Mat_<uchar> partialResult(height, width);
		Mat_<uchar> resultImg(height, width);
	
		Mat_<float> Gx(1, filterSize);
		Mat_<float> Gy(filterSize, 1);

		for (int i = 0; i < filterSize; i++) {
			float exponent1 = pow(i - filterSize, 2) / (2 * pow(filterSize / 6.0, 2));
			Gx(1, i) = exp(-exponent1) / (sqrt(2.0 * PI) * filterSize / 6.0);
			for (int j = 0; j < filterSize; j++) {
				float exponent2 = pow(j - filterSize, 2) / (2 * pow(filterSize / 6.0, 2));
				Gy(j, 1) = exp(-exponent2) / (sqrt(2.0 * PI) * filterSize / 6.0);
			}
		}

		convolutionFloat(Gx, initImg, partialResult);
		convolutionFloat(Gy, partialResult, resultImg);

		imshow("Initial image", initImg);
		imshow("Result image", resultImg);
		waitKey(0);
	}
}

//Lab 11
void cannyConvolution(Mat_<int> &filter, Mat_<uchar> initImg, Mat_<int> resultImg) {
	int height = initImg.rows;
	int width = initImg.cols;
	int dim = filter.rows;

	resultImg.create(initImg.size());
	memcpy(resultImg.data, initImg.data, height * width * sizeof(uchar));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float sum = 0;
			for (int u = 0; u < dim; u++) {
				for (int v = 0; v < dim; v++) {
					int pixelX = i + u - dim / 2;
					int pixelY = j + v - dim / 2;

					if (isInside(initImg, pixelX, pixelY)) {
						sum += filter(u, v) * initImg(pixelX, pixelY);
					}
				}
			}
			resultImg(i, j) = sum;
		}
	}
}

void computeMagnitudeAndOrientation(Mat_<float> magnitude, Mat_<float> orientation, Mat_<int> sobelXInteger, Mat_<int> sobelYInteger) {
	int height = sobelXInteger.rows;
	int width = sobelXInteger.cols;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float value = sqrt(pow(sobelXInteger(i, j), 2) + pow(sobelYInteger(i, j), 2));
			magnitude(i, j) = value;
			orientation(i, j) = atan2(sobelXInteger(i, j), sobelYInteger(i, j)) + PI;
		}
	}
}

void normalizeMagnitude(Mat_<float> magnitude, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			magnitude(i, j) = magnitude(i, j) / (4 * sqrt(2));
		}
	}
}

int getSlice(float orientation) {
	if (orientation >= 3 * PI / 8 && orientation < 5 * PI / 8 || orientation >= 11 * PI / 8 && orientation < 13 * PI / 8) {
		return 0;
	}
	else {
		if (orientation >= PI / 8 && orientation < 3 * PI / 8 || orientation >= 9 * PI / 8 && orientation < 11 * PI / 8) {
			return 1;
		}
		else {
			if (orientation >= 7 * PI / 8 && orientation < 9 * PI / 8 || orientation >= 15 * PI / 8 || orientation < PI / 8) {
				return 2;
			}
			else {
				if(orientation >= 5 * PI / 8 && orientation < 7 * PI / 8 || orientation >= 13 * PI / 8 && orientation < 15 * PI / 8) {
					return 3;
				}
			}
		}
	}
}

float getMagnitude(int slice, Mat_<float> magnitude, int i, int j) {
	float m = magnitude(i, j);
	switch (slice) {
	case 0:
		if (m <= magnitude(i - 1, j) && m <= magnitude(i + 1, j)) {
			return m;
		}
		else {
			return 0;
		}
	case 1:
		if (m > magnitude(i + 1, j - 1) && m > magnitude(i - 1, j + 1)) {
			return m;
		}
		else {
			return 0;
		}
	case 2:
		if (m > magnitude(i, j - 1) && m > magnitude(i, j + 1)) {
			return m;
		}
		else {
			return 0;
		}
	case 3:
		if (m > magnitude(i - 1, j - 1) && m > magnitude(i + 1, j + 1)) {
			return m;
		}
		else {
			return 0;
		}
	default:
		break;
	}
}

void adaptiveThresholding(Mat_<uchar> normMagCopy, int height, int width) {
	/*for (int i = 0; i < height; i++) {
		normMagCopy(i, 0) = 0;
		normMagCopy(i, width - 1) = 0;
	}
	for (int i = 0; i < width; i++) {
		normMagCopy(0, i) = 0;
		normMagCopy(height - 1, i) = 0;
	}*/

	//Compute histogram to find out no of black pixels
	//int *h = (int*)malloc(256 * sizeof(int));
	int h[256] = { 0 };
	computeHistogram(h, normMagCopy, "Histogram");

	//make the assumption that 10% of pixels which have the gradient magnitude > 0 are edge pixels
	float p = 0.1f;

	int edgePixels = p * (height * width - h[0]);
	int nonEdgePixels = (1 - p) * ((height - 2) * (width - 2) - h[0]);

	//compute high thresholding
	int sum = 0;
	int thresholdHigh = 1;
	while (thresholdHigh < 256 && sum <= nonEdgePixels) {
		sum += h[thresholdHigh];
		thresholdHigh++;
	}

	//compute low thresholding
	float thresholdLow = 0.4 * thresholdHigh;

	printf("%f %d\n", thresholdLow, thresholdHigh);

	int NON_EDGE = 0;
	int WEAK_EDGE = 128;
	int STRONG_EDGE = 255;

	Mat_<uchar> image(height, width);

	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			if (normMagCopy(i, j) < thresholdLow) {
				normMagCopy(i, j) = NON_EDGE;
			}
			else {
				if (normMagCopy(i, j) >= thresholdLow && normMagCopy(i, j) <= thresholdHigh) {
					normMagCopy(i, j) = WEAK_EDGE;
				}
				else {
					normMagCopy(i, j) = STRONG_EDGE;
				}
			}
		}
	}
}

void edgeLinking(Mat_<uchar> normMagCopy, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//STRONG_EDGE point found
			if (normMagCopy(i, j) == 255) {
				Point p(j, i);
				std::queue<Point> q;
				q.push(p);
				while (!q.empty()) {
					//Extracts the first point from the queue
					Point currentPoint = q.front();
					q.pop();
					//Find all the WEAK_EDGE neighbors of the current point
					int di[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
					int dj[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
					for (int k = 0; k < 8; k++) {
						//Find all the WEAK_EDGE neighbors of the current point
						if (normMagCopy(currentPoint.y + di[k], currentPoint.x + dj[k]) == 128) {
							//Label in the image all these neighbors as STRONG_EDGE points
							normMagCopy(currentPoint.y + di[k], currentPoint.x + dj[k]) = 255;
							//Push the image coordinates of these neighbors into the queue
							q.push(Point(currentPoint.x + dj[k], currentPoint.y + di[k]));
						}
					}
				}
			}
		}
	}

	for (int i = 2; i < height; i++) {
		for (int j = 2; j < width; j++)
		{
			if (normMagCopy(i, j) == 128)
				normMagCopy(i, j) = 0;
		}
	}
}

void cannyAlgorithm() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> initImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = initImg.rows;
		int width = initImg.cols;

		//Step 1: Filter noise with a Gaussian kernel
		Mat_<uchar> filteredImage(height, width);
		gaussianFilter2D(initImg, 3, filteredImage);

		//Step 2: Compute the gradient’s module and direction
		int sobelXData[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
		Mat_<int> sobelX(3, 3, sobelXData);

		int sobelYData[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
		Mat_<int> sobelY(3, 3, sobelYData);

		//Create int matrices by applying sobel
		Mat_<int> sobelXInteger(height, width);
		Mat_<int> sobelYInteger(height, width);
		cannyConvolution(sobelX, filteredImage, sobelXInteger);
		cannyConvolution(sobelY, filteredImage, sobelYInteger);

		//Transform int matrices to uchar to display them
		Mat_<uchar> resultSobelX(height, width);
		Mat_<uchar> resultSobelY(height, width);
		sobelXInteger.convertTo(resultSobelX, CV_8UC1);
		sobelYInteger.convertTo(resultSobelY, CV_8UC1);

		//imshow("X", resultSobelX);
		//imshow("Y", resultSobelY);
		//waitKey(0);

		//Compute the gradient module / magnitude and the gradient orientation
		//Matrics to store the magnitudes
		Mat_<float> magnitude(height, width);

		//Matrics to store the gradient orientation
		Mat_<float> orientation(height, width);

		computeMagnitudeAndOrientation(magnitude, orientation, sobelXInteger, sobelYInteger);

		//Normalize the gradient magnitude to fit in range [0-255]
		normalizeMagnitude(magnitude, height, width);

		//Normalized uchar matrix to display it.
		Mat_<uchar> normMagn(height, width);
		magnitude.convertTo(normMagn, CV_8UC1);
		//imshow("Normalized image", normMagn);
		//waitKey(0);

		//Step 3: Non-maxima suppression of the gradient’s module
		//Create a copy of the normalized matrix
		Mat_<uchar> normMagCopy(height, width);
		normMagCopy.create(normMagn.size());
		memcpy(normMagCopy.data, normMagn.data, height * width * sizeof(uchar));

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int slice = getSlice(orientation(i, j));
				normMagCopy(i, j) = getMagnitude(slice, normMagCopy, i, j);
			}
		}

		//imshow("After supression", normMagCopy);
		//waitKey(0);

		//Step 4: Edge linking through adaptive hysteresis thresholding
		//Step 4.1: Adaptive thresholding
		adaptiveThresholding(normMagCopy, height, width);
		imshow("Thresholding", normMagCopy);

		//Step 4.2: Edge linking through hysteresis
		edgeLinking(normMagCopy, height, width);

		imshow("Result", normMagCopy);
		waitKey(0);
	}
}

int main()
{
	int op;
	int addFactor;
	int mulFactor;
	int newX, newY;
	int threshold;
	bool inside;
	int n;

	//lab3
	int h[256] = { 0 };
	float p[256] = { 0 };

	Mat img = imread("Images/kids.bmp", CV_LOAD_IMAGE_UNCHANGED);
	Mat_<uchar> initImg = imread("Images/baloons.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> cameraman = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> saturn = imread("Images/saturn.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> result;
	Mat_<uchar> result1;
	Mat_<uchar> result2;

	// Lab 9
	// LOW PASS
	// mean filter 5x5
	int meanFilterData5x5[25];
	std::fill_n(meanFilterData5x5, 25, 1);
	Mat_<int> meanFilter5x5(5, 5, meanFilterData5x5);

	// mean filter 3x3
	Mat_<int> meanFilter3x3(3, 3, meanFilterData5x5);

	// mean filter 7x7
	Mat_<int> meanFilter7x7(7, 7, meanFilterData5x5);

	// gaussian filter
	int gaussianFilterData[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	Mat_<int> gaussianFilter(3, 3, gaussianFilterData);

	// HIGH PASS
	// laplace filter 3x3
	int laplaceFilterData[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
	Mat_<int> laplaceFilter(3, 3, laplaceFilterData);

	int highpassFilterData[9] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
	Mat_<int> highpassFilter(3, 3, highpassFilterData);

	//Lab 10
	int filterSize;

	do
	{
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Test resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Negative image function\n");
		printf(" 11 - Change gray level by an additive factor\n");
		printf(" 12 - Change gray level by a multiplicative factor\n");
		printf(" 13 - Create a color image with 4 squares\n");
		printf(" 14 - Horizontal flip\n");
		printf(" 15 - Vertical flip\n");
		printf(" 16 - Center crop\n");
		printf(" 17 - Resize image\n");
		printf(" 18 - Lab 2, RGB to 3 matrices\n");
		printf(" 19 - Lab 2, RGB to Grayscale\n");
		printf(" 20 - Lab 2, Grayscale to binary\n");
		printf(" 21 - Lab 2, HSV from RGB\n");
		printf(" 22 - Lab 2, is inside\n");
		printf(" 23 - Lab 3, show histogram\n");
		printf(" 24 - Lab 3, multilevel thresholding\n");
		printf(" 25 - Lab 3, floyd-steinberg\n");
		printf(" 26 - Lab 4, compute paramaters and display in standard output\n");
		printf(" 27 - Lab 5, BFS\n");
		printf(" 28 - Lab 5, Two-pass connected-component labeling\n");
		printf(" 29 - Lab 6, Border Tracing\n");
		printf(" 30 - Lab 6, Build the border from a sequence of chain codes\n");
		printf(" 31 - Lab 7, Dilation\n");
		printf(" 32 - Lab 7, Erosion\n");
		printf(" 33 - Lab 7, Opening\n");
		printf(" 34 - Lab 7, Closing\n");
		printf(" 35 - Lab 7, Boundary extraction\n");
		printf(" 36 - Lab 7, Region filling\n");
		printf(" 37 - Lab 8, Mean value and standard deviation of intensity levels\n");
		printf(" 38 - Lab 8, Thresholding algorithm\n");
		printf(" 39 - Lab 8, Histogram slide\n");
		printf(" 40 - Lab 8, Histogram stretching and shrinking\n");
		printf(" 41 - Lab 8, Gamma correction\n");
		printf(" 42 - Lab 8, Histogram equalization\n");
		printf(" 43 - Lab 9, Convolutional, low pass filter\n");
		printf(" 44 - Lab 9, Convolutional, high pass filter\n");
		printf(" 45 - Lab 9, Centered logarithm of magnitude spectra\n");
		printf(" 46 - Lab 9, Ideal low and high pass filter\n");
		printf(" 47 - Lab 9, Gaussian low and high pass filter\n");
		printf(" 48 - Lab 10, Median filter\n");
		printf(" 49 - Lab 10, 2D Gaussian filter\n");
		printf(" 50 - Lab 10, Gaussian filter using a Gaussian kernel\n");
		printf(" 51 - Lab 11, Canny algorithm\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				testNegativeImage();
				break;
			case 11:
				printf(" Additive factor: ");
				scanf("%d", &addFactor);
				changeAddGrayLevels(addFactor);
				break;
			case 12:
				printf(" Multiplicative factor: ");
				scanf("%d", &mulFactor);
				changeMulGrayLevels(mulFactor);
				break;
			case 13:
				createImage();
				break;
			case 14:
				horizontalFlip();
				break;
			case 15:
				verticalFlip();
				break;
			case 16:
				centerCrop();
				break;
			case 17:
				printf(" New x and y: ");
				scanf("%d %d", &newX, &newY);
				imageResize(newX, newY);
				break;
			case 18:
				rgbToMatrix();
				break;
			case 19:
				rgbToGrayscale();
				break;
			case 20:
				printf(" Threshold: ");
				scanf("%d", &threshold);
				grayscaleToBlackAndWhite(threshold);
				break;
			case 21:
				hsvFromRGB();
				break;
			case 22:
				printf("\n Give the positions: ");
				scanf("%d %d", &newX, &newY);
				inside = isInside(img, newX, newY);
				printf("%d\n", inside);
				break;
			case 23:
				computeHistogram(h, cameraman, "Histogram");
				break;
			case 24:
				multiLevelThresholding(p, h, 5, 0.0003f);
				break;
			case 25:
				floydSteinberg(p, h, 5, 0.0003f);
				break;
			case 26:
				computationForObject("Computations");
				break;
			case 27:
				bfs(8);
				break;
			case 28:
				twoPassLabeling();
				break;
			case 29:
				borderTracing();
				break;
			case 30:
				buildBorder();
				break;
			case 31:
				result = dilation(initImg);
				showImages(initImg, result, "Dilation");
				break;
			case 32:
				result = erosion(initImg);
				showImages(initImg, result, "Erosion");
				break;
			case 33:
				printf("Number of times to apply opening: ");
				scanf("%d", &n);
				result = openingTimes(initImg, n);
				showImages(initImg, result, "Opening");
				break;
			case 34:
				printf("Number of times to apply closing: ");
				scanf("%d", &n);
				result = closingTimes(initImg, n);
				showImages(initImg, result, "Closing");
				break;
			case 35:
				boundaryExtraction();
				break;
			case 36:
				filling();
				break;
			case 37:
				computeStandardDeviation();
				break;
			case 38:
				thresholdingAlgorithm();
				break;
			case 39:
				int offset;
				printf("Offset for the histogram slide: ");
				scanf("%d", &offset);
				histogramSlide(offset);
				break;
			case 40:
				int gOutMin, gOutMax;
				printf("Write gOutMin and gOutMax: ");
				scanf("%d %d", &gOutMin, &gOutMax);
				stretchingShrinking(gOutMin, gOutMax);
				break;
			case 41:
				float gamma;
				printf("Enter the value for gamma: ");
				scanf("%f", &gamma);
				gammaCorrection(gamma);
				break;
			case 42:
				histogramEqualization();
				break;
			case 43:
				convolution(meanFilter3x3, cameraman, result1);
				convolution(meanFilter5x5, cameraman, result2);
				imshow("Initial image", cameraman);
				imshow("3x3 filter", result1);
				imshow("5x5 filter", result2);
				waitKey(0);
				break; 
			case 44:
				convolution(gaussianFilter, cameraman, result1);
				convolution(laplaceFilter, cameraman, result2);
				convolution(highpassFilter, cameraman, result);
				imshow("Initial image", cameraman);
				imshow("Gaussian filter", result1);
				imshow("Laplace filter", result2);
				imshow("High pass filter", result);
				waitKey(0);
				break;
			case 45:
				generic_frequency_domain_filter("low_pass", cameraman);
				break;
			case 46:
				result = generic_frequency_domain_filter("low_pass", cameraman);
				result1 = generic_frequency_domain_filter("high_pass", cameraman);
				imshow("Initial image", cameraman);
				imshow("Ideal low pass filter", result);
				imshow("Ideal high pass filter", result1);
				waitKey(0);
				break;
			case 47:
				result = generic_frequency_domain_filter("gaussian_low_pass", cameraman);
				result1 = generic_frequency_domain_filter("gaussian_high_pass", cameraman);
				imshow("Initial image", cameraman);
				imshow("Gaussian low pass filter", result);
				imshow("Gaussian high pass filter", result1);
				waitKey(0);
				break;
			case 48:
				printf("Enter the variable dimension: ");
				scanf("%d", &filterSize);
				medianFilter(filterSize);
				break;
			case 49:
				printf("Enter the variable dimension: ");
				scanf("%d", &filterSize);
				gaussianFilter2D(saturn, filterSize, result);
				break;
			case 50:
				printf("Enter the variable dimension: ");
				scanf("%d", &filterSize);
				gaussianFilterUsingGaussianKernel(filterSize);
				break;
			case 51:
				cannyAlgorithm();
				break;
		}
	}
	while (op!=0);
	return 0;
}