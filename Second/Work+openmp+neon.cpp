#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <chrono>
//#include <arm_neon.h>

using namespace std;
using namespace cv;
using namespace std::chrono; // 使用chrono库

void Sobel_edge_Neon_OpenMP(Mat input_frame, Mat& output_frame); // Sobel边缘检测函数
void Hough_lines(Mat input_frame, Mat& output_frame); // Hough直线检测函数
void final(Mat input_frame, Mat hough_frame, Mat& output_frame); // 绘制原图像加上Hough直线检测所绘制的线的图像
bool lineDetector(Mat hough_frame, int threshold);

int main() {
    //图像声明
    Mat frame, sum_frame, sobel_frame, hough_frame, final_frame;
    const int num_frames = 2; // 合并帧数
    VideoCapture cap("test1.mp4");

    if (!cap.isOpened()) {
        cerr << "Error: Unable to open the video file." << endl;
        return -1;
    }
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('Y', 'U', 'Y', 'V')); // 编码格式为YUYV422
    cap.set(CAP_PROP_FRAME_WIDTH, 1280); //宽度<=1280
    cap.set(CAP_PROP_FRAME_HEIGHT, 720); //高度<=720
    cap.set(CAP_PROP_FPS, 30); //帧率 帧/秒<=30fps

    // 创建窗口并显示原始图像、合并帧图像、Sobel处理后的图像、Hough直线检测结果和最终结果
    namedWindow("Original Frame", WINDOW_NORMAL);
    namedWindow("Enhanced Frame", WINDOW_NORMAL);
    namedWindow("Sobel Frame", WINDOW_NORMAL);
    namedWindow("Hough Frame", WINDOW_NORMAL);
    namedWindow("Final Frame", WINDOW_NORMAL);
    
    while (true) {
        auto start_whole_loop = high_resolution_clock::now(); // 记录整个循环的起始时间点
      
        cap >> frame;

        if (frame.empty()) {
            break;
        }
        sum_frame = frame;
        imshow("Original Frame", frame);
        for (int i = 0; i < num_frames - 1; i++) {
            sum_frame = sum_frame + frame;
        }

        auto start_enhance = high_resolution_clock::now(); // 记录低光度处理开始时间
        imshow("Enhanced Frame", sum_frame);
        auto end_enhance = high_resolution_clock::now(); // 记录低光度处理结束时间
        auto duration_enhance = duration_cast<milliseconds>(end_enhance - start_enhance); // 计算低光度处理时间
        cout << "低光度处理时间：" << duration_enhance.count() << " ms" << endl;

        auto start_sobel = high_resolution_clock::now(); // 记录Sobel边缘检测开始时间
        Sobel_edge_Neon_OpenMP(sum_frame, sobel_frame);
        auto end_sobel = high_resolution_clock::now(); // 记录Sobel边缘检测结束时间
        auto duration_sobel = duration_cast<milliseconds>(end_sobel - start_sobel); // 计算Sobel边缘检测时间

        imshow("Sobel Frame", sobel_frame); // 在显示器上显示边缘检测图像
        cout << "Sobel边缘检测时间：" << duration_sobel.count() << " ms" << endl;

        auto start_hough = high_resolution_clock::now(); // 记录Hough直线检测开始时间
        Hough_lines(sobel_frame, hough_frame);
        auto end_hough = high_resolution_clock::now(); // 记录Hough直线检测结束时间
        auto duration_hough = duration_cast<milliseconds>(end_hough - start_hough); // 计算Hough直线检测时间

        imshow("Hough Frame", hough_frame); // 在显示器上显示Hough直线检测结果
        cout << "Hough直线检测时间：" << duration_hough.count() << " ms" << endl;

        auto start_final = high_resolution_clock::now(); // 记录final图像处理开始时间
        final(frame, hough_frame, final_frame);
        auto end_final = high_resolution_clock::now(); // 记录final图像处理结束时间
        auto duration_final = duration_cast<milliseconds>(end_final - start_final); // 计算final图像处理时间

        imshow("Final Frame", final_frame); // 在显示器上显示原图像加上Hough直线检测所绘制的线的图像
        

        cout << "final图像处理时间：" << duration_final.count() << " ms" << endl;

        bool index = lineDetector(hough_frame, 100); // 调用lineDetector函数检测车道线是否偏离
        if (index) {
            cout << "车道偏移！！注意！！！" << endl;
        } else {
            cout << "未发现车道偏移" << endl;
        }
        
        auto end_whole_loop = high_resolution_clock::now(); // 记录整个循环的结束时间点
        auto duration_whole_loop = duration_cast<milliseconds>(end_whole_loop - start_whole_loop); // 计算整个循环的时间


        cout << "整个循环时间：" << duration_whole_loop.count() << " ms" << endl;

        waitKey(30);
    }
    return 0;
}





void Sobel_edge_Neon_OpenMP(Mat input_frame, Mat& output_frame) {
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    // 求x方向梯度
    cv::Sobel(input_frame, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);

    // 求y方向梯度
    cv::Sobel(input_frame, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // 获取图像宽度和高度
    int width = input_frame.cols;
    int height = input_frame.rows;

    // 输出图像初始化
    output_frame.create(height, width, input_frame.type());
    output_frame = cv::Scalar(0, 0, 0);

    // NEON优化批量处理中间的像素（每次处理8个像素）
    int x = 0;
    //openmp优化
    #pragma omp parallel for private(x)
    for (; x <= width - 8; x += 8) {
        for (int y = 0; y < height; y++) {
            const uchar* abs_x_ptr = abs_grad_x.ptr<uchar>(y);
            const uchar* abs_y_ptr = abs_grad_y.ptr<uchar>(y);
            uchar* output_ptr = output_frame.ptr<uchar>(y);

            // 逐个通道处理像素
            for (int c = 0; c < input_frame.channels(); c++) {
                uint8x8_t x_data = vld1_u8(abs_x_ptr + (x + c) * input_frame.channels());
                uint8x8_t y_data = vld1_u8(abs_y_ptr + (x + c) * input_frame.channels());

                // 将两个寄存器中的数据相加并除以2，然后取整和饱和处理
                uint16x8_t sum = vaddl_u8(x_data, y_data);
                uint8x8_t result_u8 = vqrshrn_n_u16(sum, 1);

                // 将结果写入临时数组的对应位置
                output_ptr[x * input_frame.channels() + c] = vget_lane_u8(result_u8, 0);
                output_ptr[(x + 1) * input_frame.channels() + c] = vget_lane_u8(result_u8, 1);
                output_ptr[(x + 2) * input_frame.channels() + c] = vget_lane_u8(result_u8, 2);
                output_ptr[(x + 3) * input_frame.channels() + c] = vget_lane_u8(result_u8, 3);
                output_ptr[(x + 4) * input_frame.channels() + c] = vget_lane_u8(result_u8, 4);
                output_ptr[(x + 5) * input_frame.channels() + c] = vget_lane_u8(result_u8, 5);
                output_ptr[(x + 6) * input_frame.channels() + c] = vget_lane_u8(result_u8, 6);
                output_ptr[(x + 7) * input_frame.channels() + c] = vget_lane_u8(result_u8, 7);
            }
        }
    }

    // 处理剩余的像素（同之前，逐个像素处理）
    for (; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < input_frame.channels(); c++) {
                int index = y * width * input_frame.channels() + x * input_frame.channels() + c;
                output_frame.data[index] = (abs_grad_x.data[index] + abs_grad_y.data[index]) >> 1;
            }
        }
    }
}



void Hough_lines(Mat input_frame, Mat& output_frame) {
    Mat gray_frame, edges;
    cvtColor(input_frame, gray_frame, COLOR_BGR2GRAY);

    // 高斯模糊处理
    GaussianBlur(gray_frame, gray_frame, Size(5, 5), 0);

    // 边缘检测
    Canny(gray_frame, edges, 50, 100); // 适当降低高低阈值的差异

    // 设置区域兴趣（ROI）
    Mat mask = Mat::zeros(edges.size(), edges.type());
    Point pts[4] = {Point(0, edges.rows), Point(edges.cols / 2 - 50, edges.rows / 2 + 50),
                    Point(edges.cols / 2 + 50, edges.rows / 2 + 50), Point(edges.cols, edges.rows)};
    fillConvexPoly(mask, pts, 4, Scalar(255));
    bitwise_and(edges, mask, edges);

    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 30, 5); // 适当调整Hough直线检测参数

    cvtColor(edges, output_frame, COLOR_GRAY2BGR); // 将边缘图像转换回彩色图像

    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];

        // 计算线段的斜率
        float slope = static_cast<float>(l[3] - l[1]) / static_cast<float>(l[2] - l[0]);

        // 设置角度范围，只保留接近水平或垂直的线段
        if (abs(slope) > 0.3 && abs(slope) < 3) {
            line(output_frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
        }
    }
}


void final(Mat input_frame, Mat hough_frame, Mat& output_frame) {
    // 创建final_frame并将原图像复制到其中
    output_frame = input_frame.clone();

    // 将Hough直线检测所绘制的线的图像添加到final_frame中
    for (int y = 0; y < hough_frame.rows; y++) {
        for (int x = 0; x < hough_frame.cols; x++) {
            Vec3b color = hough_frame.at<Vec3b>(y, x);
            if (color[0] == 0 && color[1] == 0 && color[2] == 255) {
                output_frame.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
            }
        }
    }
}

bool lineDetector(Mat hough_frame, int threshold) {
    int left_count = 0;  // 记录左侧车道线数量
    int right_count = 0; // 记录右侧车道线数量

    int mid_x = hough_frame.cols / 2; // 图像中心点的x坐标

    // 遍历Hough直线检测结果
    for (int i = 0; i < hough_frame.rows; i++) {
        for (int j = 0; j < hough_frame.cols; j++) {
            Vec3b pixel = hough_frame.at<Vec3b>(i, j);
            // 如果是红色像素点，表示检测到车道线
            if (pixel[2] > 0 && pixel[1] == 0 && pixel[0] == 0) {
                // 判断车道线在图像的左侧还是右侧
                if (j < mid_x) {
                    left_count++;
                } else {
                    right_count++;
                }
            }
        }
    }

    // 判断车道线是否偏离
    if (left_count < threshold && right_count < threshold) {
        return true; // 左右两侧车道线数量都小于阈值，表示车道偏离
    } else {
        return false; // 车道线数量满足要求，表示车道未偏离
    }
}