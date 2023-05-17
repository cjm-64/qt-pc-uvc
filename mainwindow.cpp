#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "globals.h"

#include "stdio.h"
#include "iostream"
#include "stdio.h"
#include "unistd.h"
#include "string"
#include "libuvc/libuvc.h"
#include "turbojpeg.h"
#include "math.h"
#include "chrono"
#include "list"
#include "fstream"

// Computer Vision
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;


//Data strucures
struct CamSettings{
    string CamName;
    int width;
    int height;
    int fps;
};

// ID0 = right, ID1 = Left, ID2 = World
struct CamInfo{
    string CamName;
    uint8_t vID;
    uint8_t pID;
    string uid;
    int cam_num;
};

struct StreamingInfo{
    uvc_context_t *ctx;
    uvc_device_t *dev;
    uvc_device_handle_t *devh;
    uvc_stream_ctrl_t ctrl;
    uvc_stream_handle_t *strmh;
    const uvc_format_desc_t *format_desc;
    uvc_frame_desc_t *frame_desc;
    enum uvc_frame_format frame_format;
};

struct FrameProcessingInfo{
    int thresh_val; //Threshold value for gray to binary conversion
    int max_radius; //Max radius circle to search for in HoughCircles()
};

struct PositionData{
    int X_Pos;
    int Y_Pos;
    int Radius;
};

//Instantiate global data strucutres
CamSettings camSets[2];
CamInfo Cameras[2];
StreamingInfo CamStreams[2];
FrameProcessingInfo FrameProc[2];

//Other global declarations
Scalar col = Scalar(0, 255, 0); //Color for drawing on frame
int save_placeholder[4] = {0};
string headers[3][1] = {{"Right Calibration"}, {"Left Calibration"}, {"Test Data"}};
int step = 1;
int headerwritten = 0;

ofstream Output_file;


//Functions to set up cameras
void getCamInfo(struct CamInfo *ci){
    uvc_context_t *ctx;
    uvc_device_t **device_list;
    uvc_device_t *dev;
    uvc_device_descriptor_t *desc;
    uvc_error_t res;

    res = uvc_init(&ctx,NULL);
    if(res < 0) {
        uvc_perror(res, "uvc_init");
    }

    res = uvc_get_device_list(ctx,&device_list);
    if (res < 0) {
        uvc_perror(res, "uvc_find_device");
    }
    int i = 0;
    int dev_struct_count = 0;
    while (true) {
        cout << i << endl;
        dev = device_list[i];
        if (dev == NULL) {
            break;
        }
        else{
            uvc_get_device_descriptor(dev, &desc);
            if(desc->idVendor != 3141){
                printf("World Cam\n");
                i++;
                continue;
            }
            else{
                ci[dev_struct_count].CamName = desc->product;
                ci[dev_struct_count].vID = desc->idVendor;
                ci[dev_struct_count].pID = desc->idProduct;
                ci[dev_struct_count].uid = to_string(uvc_get_device_address(dev))+":"+to_string(uvc_get_bus_number(dev));
                ci[dev_struct_count].cam_num = i;
                dev_struct_count++;
            }
            printf("Got desc\n");
        }
        uvc_free_device_descriptor(desc);
        i++;
    }
    uvc_exit(ctx);
}

void setUpStreams(struct CamSettings *cs, struct CamInfo *ci, struct StreamingInfo *si){
    uvc_error_t res;
    uvc_device_t **devicelist;

    for(int i = 0; i<2;i++){
        res = uvc_init(&si[i].ctx, NULL);
        if (res < 0) {
            uvc_perror(res, "uvc_init");
        }
        else{
            printf("UVC %d open success\n", i);
        }
        res = uvc_find_devices(si[i].ctx, &devicelist, 0, 0, NULL);
        if (res < 0) {
            uvc_perror(res, "uvc_init");
        }
        else{
            cout << "Dev " << i << ": " << si[i].dev << endl;
        }
        res = uvc_open(devicelist[ci[i].cam_num], &si[i].devh, 1);
        if (res < 0) {
            uvc_perror(res, "uvc_find_device"); /* no devices found */
        }
        else{
            cout << "devh " << i << ": " << si[i].devh << endl;
        }
        si[i].format_desc = uvc_get_format_descs(si[i].devh);
        si[i].frame_desc = si[i].format_desc->frame_descs->next;
        si[i].frame_format = UVC_FRAME_FORMAT_ANY;
        if(si[i].frame_desc->wWidth != NULL){
            cs[i].width = si[i].frame_desc->wWidth;
            cs[i].height = si[i].frame_desc->wHeight;
            cs[i].fps = 10000000 / si[i].frame_desc->intervals[2];
        }
        printf("\nEye %d format: (%4s) %dx%d %dfps\n\n", i, si[i].format_desc->fourccFormat, cs[i].width, cs[i].height, cs[i].fps);

        res = uvc_get_stream_ctrl_format_size(si[i].devh, &si[i].ctrl, si[i].frame_format, cs[i].width, cs[i].height, cs[i].fps, 1);
        if (res < 0){
            uvc_perror(res, "start_streaming");
        }
        else{
            printf("Eye %d stream control formatted\n", i);
            uvc_print_stream_ctrl(&si[i].ctrl, stderr);
        }

        res = uvc_stream_open_ctrl(si[i].devh, &si[i].strmh, &si[i].ctrl,1);
        if (res < 0){
            uvc_perror(res, "start_streaming");
        }
        else{
            printf("Eye %d stream opened\n", i);
        }

        res = uvc_stream_start(si[i].strmh, nullptr, nullptr,2.0,0);
        if (res < 0){
            uvc_perror(res, "start_streaming");
        }
        else{
            printf("Eye %d stream started\n", i);
        }
        printf("\n\n\n");
    }
}

void MainWindow::initCams(){
    for(int i = 0; i<2;i++){
        if(i == 0){
            camSets[i].CamName = "Pupil Cam2 ID0";
        }
        else{
            camSets[i].CamName = "Pupil Cam2 ID1";
        }
        camSets[i].width = 192;
        camSets[i].height = 192;
        camSets[i].fps = 120;
    }

    getCamInfo(Cameras);
    for (int j = 0; j<2; j++){
        cout << "Cam " << j << ": " << Cameras[j].CamName << " " << Cameras[j].uid << " " << Cameras[j].vID << " " << Cameras[j].pID << endl;
    }

    setUpStreams(camSets, Cameras, CamStreams);

}

void MainWindow::initFrameProc(){
    //Set the inital values for the frame proc struct
    for (int i=0; i<2; i++){
        FrameProc[i].thresh_val = 50;
        FrameProc[i].max_radius = 50;
    }
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    timer = new QTimer(this);

    initCams();
    initFrameProc();


    openCamera();
}

void writeToFile(int data[4]){
    if (headerwritten == 0){
        if (step == 1){
            //Right Cal so put header
            Output_file << headers[0][0] << endl;
        }
        else if (step == 2){
            //Left Cal so put header
            Output_file << headers[1][0] << endl;
        }
        else if (step == 3){
            //Testing so put header
            Output_file << headers[2][0] << endl;
        }
        else{
            printf("How did you get here?\n");
            cout << "Step: " << step << endl;
        }
        headerwritten = 1;
    }
    Output_file << save_placeholder[0] << " " << save_placeholder[1] << " " << save_placeholder[2] << " " << save_placeholder[3] << " " << endl;
}

void savePositions(PositionData &pd, int eye){
    if (eye == 0){
        //Right eye, so fill start of matrix
        save_placeholder[0] = pd.X_Pos;
        save_placeholder[1] = pd.Y_Pos;
    }
    else{
        //Left Eye, so fill end of matrix
        save_placeholder[2] = pd.X_Pos;
        save_placeholder[3] = pd.Y_Pos;
        cout << save_placeholder[0] << " " << save_placeholder[1] << " " << save_placeholder[2] << " " << save_placeholder[3] << " " << endl;
        writeToFile(save_placeholder);
    }
}

void MainWindow::openCamera(){
    qDebug() << "Open Stream";
    connect(timer, SIGNAL(timeout()), this, SLOT(updateFrame()));
    qDebug() << "Slot Connected";
    timer->start(20);
    qDebug() << "Timer Started";
}

void MainWindow::updateFrame(){

    qDebug() << "UF";

    uvc_frame_t *frame;
    uvc_frame_t *bgr;
    uvc_error_t res;

    Mat image;

    for (int i = 0; i<2;i++){
        //As a side note, the images are greyscaled, however they are treated as color when they come into the code

        res = uvc_stream_get_frame(CamStreams[i].strmh, &frame, 1* pow(10,6));
        if(res < 0){
            uvc_perror(res, "Failed to get frame");
            continue;
        }
        else{
            printf("got frame");
        }

        //Allocate buffers for conversions
        int frameW = frame->width;
        int frameH = frame->height;
        long unsigned int frameBytes = frame->data_bytes;

        printf("Eye %d: frame_format = %d, width = %d, height = %d, length = %lu\n", i, frame->frame_format, frameW, frameH, frameBytes);

        bgr = uvc_allocate_frame(frameW * frameH * 3);
        if (!bgr) {
            printf("unable to allocate bgr frame!\n");
            return;
        }
        uvc_error_t res = uvc_yuyv2bgr(frame, bgr);
        if (res < 0){
            printf("Unable to copy frame to bgr!\n");
        }
        Mat placeholder(bgr->height, bgr->width, CV_8UC3, bgr->data); //Create MAT and fill with frame data
        Mat flipped;
        placeholder.copyTo(flipped); //Copy frame to new MAT. idk why you need to do this but you do ¯\_(ツ)_/¯

        flip(flipped, image, 0); //Flip image

        Mat grayIMG, binaryIMG, bpcIMG; //Create new Mats to to image processing steps
        cvtColor(image, grayIMG, COLOR_BGR2GRAY); //Convert to grayscale
        threshold(grayIMG, binaryIMG, FrameProc[i].thresh_val, thresh_max_val, thresh_type); //Convert to binary based on thresh; controlled by slider
        cvtColor(binaryIMG, bpcIMG, COLOR_GRAY2RGB); // enable color on binary so we can draw on it later

        PositionData pd;
        vector<Vec3f> circles;
        HoughCircles(binaryIMG, circles, HOUGH_GRADIENT, 1, 1000, CED, Cent_D, FrameProc[i].max_radius-2, FrameProc[i].max_radius);
        Vec3i c;
        for( size_t i = 0; i < circles.size(); i++ ){
            c = circles[i];
        }

        pd.X_Pos = c[0]+X;
        pd.Y_Pos = c[1]+Y;
        pd.Radius = c[2];

        //savePositions(pd, i);

        //Draw Circles on Black and White
        circle(bpcIMG, Point(pd.X_Pos, pd.Y_Pos), 1, col,1,LINE_8);
        circle(bpcIMG, Point(pd.X_Pos, pd.Y_Pos), pd.Radius, col,1,LINE_8);

        Mat final_image;
        //Display Image
        //Check for which eye and if grey or Black and White (binary)
        if (i == 0){
            if (ColorOrBW == 0){
                //Right Eye Color frame
                image.copyTo(final_image);
            }
            else{
                //Right Eye BW frame
                bpcIMG.copyTo(final_image);
            }
            ui->RightEye->setPixmap(QPixmap::fromImage(QImage((unsigned char*) final_image.data, final_image.cols, final_image.rows, final_image.step, QImage::Format_RGB888)));
        }
        else{
            if (ColorOrBW == 0){
                //Right Eye Color frame
                image.copyTo(final_image);
            }
            else{
                //Right Eye BW frame
                bpcIMG.copyTo(final_image);
            }
            ui->LeftEye->setPixmap(QPixmap::fromImage(QImage((unsigned char*) final_image.data, final_image.cols, final_image.rows, final_image.step, QImage::Format_RGB888)));
        }

        //Free memory
        flipped.release();
        image.release();
        placeholder.release();
        grayIMG.release();
        binaryIMG.release();
        bpcIMG.release();
        uvc_free_frame(bgr);
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    timer->stop();
    for(int i = 0; i<2; i++){
        uvc_exit(CamStreams[i].ctx);
        cout << "Closed Cam " << i << endl;
    }

    close();
}

