#include "gstnvdsmeta.h"
#include <cuda_runtime_api.h>
#include <glib.h>
#include <gst/gst.h>
#include <sys/time.h>

#ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#endif

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/videoio.hpp>

#include "gstnvdsinfer.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include <bits/stdc++.h>
#include <opencv2/cudaarithm.hpp>
#include "FaceRecognition_DS_Final/alignment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "yaml-cpp/yaml.h"
// #include "FaceRecognition_DS_Final/ExtractFeature.h"
#include <nvdsmeta_schema.h>

#define MAX_DISPLAY_LEN 1000
// #define MAX_TRACKING_ID_LEN 16

#define PGIE_CLASS_ID_FACE 0
#define PGIE_CLASS_ID_FEATURE 1

#define DETECTION 1
#define EXTRACTION 2

#define TILED_OUTPUT_WIDTH 1920
#define TILED_OUTPUT_HEIGHT 1080
#define DELAY_VALUE 1000000
#define OSD_PROCESS_MODE 1
#define OSD_DISPLAY_TEXT 0
#define MUXER_BATCH_TIMEOUT_USEC 4000000
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
#define CHECK_ERROR(error)                                               \
  if (error)                                                             \
  {                                                                      \
    g_printerr("Error while parsing config file: %s\n", error->message); \
    goto done;                                                           \
  }
#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GPU_ID "gpu-id"
#define CUDA_DEVICE 0 // GPU id
#define FPS_PRINT_INTERVAL 300
#define MAX_TIME_STAMP_LEN 32

#include <random>
#define PGIE_CONFIG_FILE \
  "../apps/FaceRecognition_DS_Final/configs/infer_config_scrfd.txt"

#define TRACKER_CONFIG_FILE \
  "../apps/FaceRecognition_DS_Final/configs/dstest2_tracker_config.txt"
#define SGIE_CONFIG_FILE \
  "../apps/FaceRecognition_DS_Final/configs/infer_config_iresnet.txt"
#define MSCONV_CONFIG_FILE "../apps/FaceRecognition_DS_Final/configs/msgconv_config.txt"


struct Landmark
{
  float x[5];
  float y[5];
};

struct Face
{
  float x; /// Top left x
  float y; /// Top left y
  float w; /// Bounding box width
  float h; /// Bounding box height
  // float confidence;        /// Confidence of face
  // float wearing_mask_prob; /// Probability of wearing a mask
  // Landmark landmark;       /// Keypoints
  int id;
  cv::Mat image_crop;
  bool face_mask;
  std::string feature;
  std::string path_image;
  std::string path_image_full;
};



static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);

static GstPadProbeReturn pgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);
static gchar *get_absolute_file_path(gchar *cfg_file_path, gchar *file_path);
static gboolean set_tracker_properties(GstElement *nvtracker);
int FaceRecognition_DS_Final(int num_sources, char **source_paths, YAML::Node config);