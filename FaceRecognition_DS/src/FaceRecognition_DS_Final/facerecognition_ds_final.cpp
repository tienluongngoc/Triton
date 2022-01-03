#include "FaceRecognition_DS_Final/facerecognition_ds_final.h"
#include "yaml-cpp/yaml.h"

static gchar *cfg_file = NULL;
static gchar *topic = "Face";
std::string config_file = "../config.yaml";

// static gchar *conn_str = "192.168.100.126;9092;Face"; #rabbit
// static gchar *proto_lib =
// "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so";

static gchar *conn_str ;//= "localhost;5672;guest;guest";
static gchar *proto_lib = "/opt/nvidia/deepstream/deepstream/lib/libnvds_amqp_proto.so";
int frame_number = 0;
// std::map<int, Face> facedt_map;


static gint schema_type = 0;
static gboolean display_off = FALSE;
// std::string path_image_full_dir = "/home/haobk/Mydata/FaceRecognition_DS/build/data/data_full/";
// std::string path_image_crop_dir = "/home/haobk/Mydata/FaceRecognition_DS/build/data/data_crop/";
// std::string path_image_aligned_dir = "/home/haobk/Mydata/FaceRecognition_DS/build/data/data_aligned/";

GOptionEntry entries[] = {
    {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME, &cfg_file,
     "Set the adaptor config file. Optional if connection string has relevant  "
     "details.",
     NULL},
    {"topic", 't', 0, G_OPTION_ARG_STRING, &topic,
     "Name of message topic. Optional if it is part of connection string or "
     "config file.",
     NULL},
    {"conn-str", 0, 0, G_OPTION_ARG_STRING, &conn_str,
     "Connection string of backend server. Optional if it is part of config "
     "file.",
     NULL},
    {"proto-lib", 'p', 0, G_OPTION_ARG_STRING, &proto_lib,
     "Absolute path of adaptor library", NULL},
    {"schema", 's', 0, G_OPTION_ARG_INT, &schema_type,
     "Type of message schema (0=Full, 1=minimal), default=0", NULL},
    {"no-display", 0, 0, G_OPTION_ARG_NONE, &display_off, "Disable display",
     NULL},
    {NULL}};

static void print_facedt(Face facedt) {
  std::cout << "face id : " << facedt.id << " x : " << facedt.x
            << " y : " << facedt.y << " w : " << facedt.w << " h : " << facedt.h
            << "\n";
}

static std::string generate_uuid(size_t len = 10) {
  static const char x[] = "0123456789abcdef";

  std::string uuid;
  uuid.reserve(len);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(0, sizeof(x) - 2);
  for (size_t i = 0; i < len; i++)
    uuid += x[dis(gen)];

  return uuid;
}

static void generate_ts_rfc3339(char *buf, int buf_size) {
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6]; //.nnnZ\0

  clock_gettime(CLOCK_REALTIME, &ts);
  memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
  gmtime_r(&tloc, &tm_log);
  strftime(buf, buf_size, "%Y-%m-%d %H:%M:%S", &tm_log);
  // int ms = ts.tv_nsec / 1000000;
  // g_snprintf(strmsec, sizeof(strmsec), ".%.3dZ", ms);
  // strncat(buf, strmsec, buf_size);
}

static gpointer meta_copy_func(gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;

  dstMeta = static_cast<NvDsEventMsgMeta *>(
      g_memdup(srcMeta, sizeof(NvDsEventMsgMeta)));

  if (srcMeta->ts)
    dstMeta->ts = g_strdup(srcMeta->ts);

  if (srcMeta->sensorStr)
    dstMeta->sensorStr = g_strdup(srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature = static_cast<gdouble *>(
        g_memdup(srcMeta->objSignature.signature, srcMeta->objSignature.size));
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if (srcMeta->objectId) {
    dstMeta->objectId = g_strdup(srcMeta->objectId);
  }

  if (srcMeta->extMsgSize > 0) {

    NvDsPersonObject *srcObj = (NvDsPersonObject *)srcMeta->extMsg;
    NvDsPersonObject *obj =
        (NvDsPersonObject *)g_malloc0(sizeof(NvDsPersonObject));

    obj->age = srcObj->age;

    if (srcObj->gender)
      obj->gender = g_strdup(srcObj->gender);
    if (srcObj->cap)
      obj->cap = g_strdup(srcObj->cap);
    if (srcObj->hair)
      obj->hair = g_strdup(srcObj->hair);
    if (srcObj->apparel)
      obj->apparel = g_strdup(srcObj->apparel);
    dstMeta->extMsg = obj;
    dstMeta->extMsgSize = sizeof(NvDsPersonObject);
  }

  return dstMeta;
}

static void meta_free_func(gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;

  g_free(srcMeta->ts);
  g_free(srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    g_free(srcMeta->objSignature.signature);
    srcMeta->objSignature.size = 0;
  }

  if (srcMeta->objectId) {
    g_free(srcMeta->objectId);
  }

  if (srcMeta->extMsgSize > 0) {

    NvDsPersonObject *obj = (NvDsPersonObject *)srcMeta->extMsg;

    if (obj->gender)
      g_free(obj->gender);
    if (obj->cap)
      g_free(obj->cap);
    if (obj->hair)
      g_free(obj->hair);
    if (obj->apparel)
      g_free(obj->apparel);
    g_free(srcMeta->extMsg);
    srcMeta->extMsgSize = 0;
  }
  g_free(user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
}

static void generate_person_meta(gpointer data, Face facedt) {
  NvDsPersonObject *obj = (NvDsPersonObject *)data;

  // obj->cap = g_strdup(facedt.path_image.c_str());
  obj->hair = g_strdup(facedt.feature.c_str());
  if (facedt.face_mask)
    obj->gender = g_strdup("face_mask");
  else
    obj->gender = g_strdup("non_face_mask");
  // obj->apparel = g_strdup(facedt.path_image_full.c_str());
}

static void generate_event_msg_meta(gpointer data, gint class_id,
                                    NvDsObjectMeta *obj_params, Face facedt) {
  NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *)data;
  meta->bbox.top = facedt.y;
  meta->bbox.left = facedt.x;
  meta->bbox.width = facedt.w;
  meta->bbox.height = facedt.h;
  meta->frameId = frame_number;
  meta->confidence = 1;

  meta->ts = (gchar *)g_malloc0(MAX_TIME_STAMP_LEN + 1);
  meta->objectId = (gchar *)g_malloc0(MAX_LABEL_SIZE);
  strncpy(meta->objectId, obj_params->obj_label, MAX_LABEL_SIZE);
  generate_ts_rfc3339(meta->ts, MAX_TIME_STAMP_LEN);
  meta->type = NVDS_EVENT_ENTRY;
  meta->objType = NVDS_OBJECT_TYPE_PERSON;
  meta->objClassId = 0;

  NvDsPersonObject *obj =
      (NvDsPersonObject *)g_malloc0(sizeof(NvDsPersonObject));
  generate_person_meta(obj, facedt);

  meta->extMsg = obj;
  meta->extMsgSize = sizeof(NvDsPersonObject);
}

GstPadProbeReturn pgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                                        gpointer u_data) {

  // GstBuffer *buf = (GstBuffer *)info->data;
  // NvDsObjectMeta *obj_meta = NULL;
  // NvDsMetaList *l_frame = NULL;
  // NvDsMetaList *l_obj = NULL;
  // GstMapInfo in_map_info;
  // NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  // NvBufSurface *surface = NULL;
  // cv::Mat cpu_mat;

  // // // ===============================get infor object
  // // =========================
  // memset(&in_map_info, 0, sizeof(in_map_info));

  // if (gst_buffer_map(buf, &in_map_info, GST_MAP_READWRITE))
  // {

  //   for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
  //        l_frame = l_frame->next)
  //   {
  //     NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

  //     if (frame_meta == NULL)
  //     {
  //       continue;
  //     }
  //     if (frame_meta->obj_meta_list != NULL)
  //     {
  //       surface = (NvBufSurface *)in_map_info.data;
  //       void *data_ptr = surface->surfaceList[frame_meta->batch_id].dataPtr;
  //       int src_height = surface->surfaceList[frame_meta->batch_id].height;
  //       int src_width = surface->surfaceList[frame_meta->batch_id].width;
  //       int data_size = surface->surfaceList[frame_meta->batch_id].dataSize;
  //       uint32_t pitch = surface->surfaceList[frame_meta->batch_id].pitch;
  //       cv::cuda::GpuMat gpu_mat = cv::cuda::GpuMat(src_height, src_width, CV_8UC4, data_ptr);
  //       gpu_mat.download(cpu_mat);
  //       cv::cvtColor(cpu_mat, cpu_mat, CV_RGBA2BGRA);
  //       std::string path_image_full = path_image_full_dir + generate_uuid() + ".jpg";
  //       cv::Mat image_save;
  //       cv::resize(cpu_mat, image_save, size_image_save, 0, 0, CV_INTER_LINEAR);
  //       cv::imwrite(path_image_full, image_save);

  //       // cv::imwrite(path_image_full, cpu_mat);

  //       // cv::cuda::GpuMat gpu_mat = cv::cuda::GpuMat(src_height * 1.5,
  //       //                                             src_width,
  //       //                                             CV_8UC1, data_ptr,
  //       //                                             pitch);
  //       // gpu_mat.download(cpu_mat);
  //       // cv::cvtColor(cpu_mat, cpu_mat, cv::COLOR_YUV2BGR_NV12);
  //       // cv::imwrite(std::to_string(d) + "image.jpg", cpu_mat);
  //       // d += 1;
  //       int d = 0 ;
  //       for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
  //            l_obj = l_obj->next)
  //       {

  //         obj_meta = (NvDsObjectMeta *)(l_obj->data);
  //         if (obj_meta == NULL)
  //           continue;

  //         // Face facedt;
  //         if (obj_meta->unique_component_id == 1)
  //         {
  //           if ((float)obj_meta->mask_params.size > 0)
  //           {

  //             // facedt.x = obj_meta->rect_params.left;
  //             // facedt.y = obj_meta->rect_params.top;
  //             // facedt.w = obj_meta->rect_params.width;
  //             // facedt.h = obj_meta->rect_params.height;
  //             // facedt.id = obj_meta->object_id + 1;

  //             float x = obj_meta->rect_params.left;
  //             float y = obj_meta->rect_params.top;
  //             float w = obj_meta->rect_params.width;
  //             float h = obj_meta->rect_params.height;
  //             int id = obj_meta->object_id + 1;

  //             // obj_meta->text_params.display_text=obj_meta->text_params.display_text+"abc".c_str();
  //             // facedt.path_image_full = path_image_full;
  //             cv::Mat image_crop;
  //             cv::Rect myROI(x, y, w, h);
  //             cpu_mat(myROI).copyTo(image_crop);
  //             std::string path_image = path_image_crop_dir + std::to_string(id) + "_" + generate_uuid() + "_img_cropped.jpg";
  //             cv::imwrite(path_image, image_crop);
  //             std::string s = std::string(obj_meta->text_params.display_text) + " " + path_image + " " + path_image_full;

  //             cv::Mat image_alinged;
  //             cv::Rect myROI1(112*d, 0, 112,112);
  //             d+=1;
  //             cpu_mat(myROI1).copyTo(image_alinged);
  //             std::string path_image_aligned = path_image_aligned_dir + std::to_string(id) + "_" + generate_uuid() + "_img_aligned.jpg";
  //             cv::imwrite(path_image_aligned, image_alinged);
         

              

  //             obj_meta->text_params.display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
  //             snprintf(obj_meta->text_params.display_text, MAX_DISPLAY_LEN, s.c_str());

  //             // facedt_map[facedt.id] = facedt;
  //           }
  //         }
  //       }
  //     }
  //   }
  //   gst_buffer_unmap(buf, &in_map_info);
  // }
  // else
  // {
  //   std::cout << "fail\n";
  // }

  // return GST_PAD_PROBE_OK;
}

GstPadProbeReturn tiler_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                                             gpointer u_data) {
  gchar *msg = NULL;
  g_object_get(G_OBJECT(u_data), "last-message", &msg, NULL);
  if (msg != NULL) {
    g_print("Fps info: %s\n", msg);
  }

  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  GstMapInfo in_map_info;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  NvBufSurface *surface = NULL;

  // // ===============================get infor object
  // =========================

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

    if (frame_meta == NULL) {
      continue;
    }
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {

      obj_meta = (NvDsObjectMeta *)(l_obj->data);
      if (obj_meta == NULL) {
        // Ignore Null object.
        continue;
      }
      Face facedt;
      
      if (obj_meta->unique_component_id == 1) {

        if ((float)obj_meta->mask_params.size > 0)
        {
          int id = obj_meta->object_id + 1;
          facedt.id = id;

          // std::cout << "text_params.display_text " << obj_meta->text_params.display_text << "\n";
          obj_meta->rect_params.left = obj_meta->mask_params.data[11] * 3;
          obj_meta->rect_params.top = obj_meta->mask_params.data[10] * 3;
          obj_meta->rect_params.width = obj_meta->mask_params.data[13] * 3 - obj_meta->rect_params.left;
          obj_meta->rect_params.height = obj_meta->mask_params.data[12] * 3 - obj_meta->rect_params.top;

          facedt.x = obj_meta->rect_params.left;
          facedt.y = obj_meta->rect_params.top;
          facedt.w = obj_meta->rect_params.width;
          facedt.h = obj_meta->rect_params.height;

          std::string result_str = obj_meta->text_params.display_text;

          if (result_str[0] == '1')
            facedt.face_mask = true;
          else
            facedt.face_mask = false;

          // print_facedt(facedt);
          facedt.feature = result_str;

          // add kafka ....................................................................
          NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
          msg_meta->sensorId = frame_meta->source_id;
          generate_event_msg_meta(msg_meta, obj_meta->class_id, obj_meta,
                                  facedt);

          NvDsUserMeta *user_event_meta =
              nvds_acquire_user_meta_from_pool(batch_meta);
          if (user_event_meta) {
            user_event_meta->user_meta_data = (void *)msg_meta;
            user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
            user_event_meta->base_meta.copy_func =
                (NvDsMetaCopyFunc)meta_copy_func;
            user_event_meta->base_meta.release_func =
                (NvDsMetaReleaseFunc)meta_free_func;
            nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
          } else {
            g_print("Error in attaching event meta to buffer\n");
          }
          // facedt_map.erase(id);
        }
      }
    }
  }

  frame_number++;

  return GST_PAD_PROBE_OK;
}

static void fps_measurements_callback(GstElement fpsdisplaysink, gdouble fps,
                                      gdouble droprate, gdouble avgfps,
                                      gpointer udata) {
  g_print("fps=%.1f droprate=%.2f avgfps=%.1f\n", fps, droprate, avgfps);
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg)) {
  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
  case GST_MESSAGE_ERROR: {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src),
               error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }
  default:
    break;
  }
  return TRUE;
}

static gchar *get_absolute_file_path(gchar *cfg_file_path, gchar *file_path) {
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path && file_path[0] == '/') {
    return file_path;
  }

  if (!realpath(cfg_file_path, abs_cfg_path)) {
    g_free(file_path);
    return NULL;
  }

  // Return absolute path of config file if file_path is NULL.
  if (!file_path) {
    abs_file_path = g_strdup(abs_cfg_path);
    return abs_file_path;
  }

  delim = g_strrstr(abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat(abs_cfg_path, file_path, NULL);
  g_free(file_path);

  return abs_file_path;
}

static gboolean set_tracker_properties(GstElement *nvtracker) {
  gboolean ret = FALSE;
  GError *error = NULL;
  gchar **keys = NULL;
  gchar **key = NULL;
  GKeyFile *key_file = g_key_file_new();

  if (!g_key_file_load_from_file(key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
                                 &error)) {
    g_printerr("Failed to load config file: %s\n", error->message);
    return FALSE;
  }

  keys = g_key_file_get_keys(key_file, CONFIG_GROUP_TRACKER, NULL, &error);
  CHECK_ERROR(error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_WIDTH)) {
      gint width = g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                          CONFIG_GROUP_TRACKER_WIDTH, &error);
      CHECK_ERROR(error);
      g_object_set(G_OBJECT(nvtracker), "tracker-width", width, NULL);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
      gint height = g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                           CONFIG_GROUP_TRACKER_HEIGHT, &error);
      CHECK_ERROR(error);
      g_object_set(G_OBJECT(nvtracker), "tracker-height", height, NULL);
    } else if (!g_strcmp0(*key, CONFIG_GPU_ID)) {
      guint gpu_id = g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                            CONFIG_GPU_ID, &error);
      CHECK_ERROR(error);
      g_object_set(G_OBJECT(nvtracker), "gpu_id", gpu_id, NULL);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
      char *ll_config_file = get_absolute_file_path(
          TRACKER_CONFIG_FILE,
          g_key_file_get_string(key_file, CONFIG_GROUP_TRACKER,
                                CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
      CHECK_ERROR(error);
      g_object_set(G_OBJECT(nvtracker), "ll-config-file", ll_config_file, NULL);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
      char *ll_lib_file = get_absolute_file_path(
          TRACKER_CONFIG_FILE,
          g_key_file_get_string(key_file, CONFIG_GROUP_TRACKER,
                                CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
      CHECK_ERROR(error);
      g_object_set(G_OBJECT(nvtracker), "ll-lib-file", ll_lib_file, NULL);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
      gboolean enable_batch_process = g_key_file_get_integer(
          key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
      CHECK_ERROR(error);
      g_object_set(G_OBJECT(nvtracker), "enable-batch-process",
                   enable_batch_process, NULL);
    } else {
      g_printerr("Unknown key '%s' for group [%s]", *key, CONFIG_GROUP_TRACKER);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free(error);
  }
  if (keys) {
    g_strfreev(keys);
  }
  if (!ret) {
    g_printerr("%s failed", __func__);
  }
  return ret;
}

static void cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad,
                      gpointer data) {
  g_print("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);
  GstElement *source_bin = (GstElement *)data;
  GstCapsFeatures *features = gst_caps_get_features(caps, 0);
  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp(name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
      if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                                    decoder_src_pad)) {
        g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref(bin_ghost_pad);
    } else {
      g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}
static void decodebin_child_added(GstChildProxy *child_proxy, GObject *object,
                                  gchar *name, gpointer user_data) {
  g_print("Decodebin child added: %s\n", name);
  if (g_strrstr(name, "decodebin") == name) {
    g_signal_connect(G_OBJECT(object), "child-added",
                     G_CALLBACK(decodebin_child_added), user_data);
  }
  if (g_strstr_len(name, -1, "nvv4l2decoder0") == name) {
    g_print("Seting bufapi_version: %s\n", name);
    // g_object_set(object, "nvbuf-memory-type", 3, NULL);

    // gboolean src_res = g_str_has_suffix(name, "0");
    // if (!src_res)
    // {
    //   g_object_set(object, "drop-frame-interval", 1, NULL);
    // }
  }
}

static GstElement *create_source_bin(guint index, gchar *uri) {
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = {};

  g_snprintf(bin_name, 15, "source-bin-%02d", index);
  bin = gst_bin_new(bin_name);
  uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr("One element in source bin could not be created.\n");
    return NULL;
  }
  g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);
  g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added", G_CALLBACK(cb_newpad),
                   bin);
  g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                   G_CALLBACK(decodebin_child_added), bin);

  gst_bin_add(GST_BIN(bin), uri_decode_bin);
  if (!gst_element_add_pad(bin,
                           gst_ghost_pad_new_no_target("src", GST_PAD_SRC))) {
    g_printerr("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

int FaceRecognition_DS_Final(int num_sources, char **sources,
                            YAML::Node config)
{
  std::string conn = config["RabbitMQ"]["conn_str"].as<std::string>().c_str();
  conn_str= new char[conn.length() + 1];
  strcpy(conn_str, conn.c_str());
  
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
             *nvvidconv = NULL, *nvosd = NULL, *nvtracker = NULL,
             *q_nvtracker = NULL, *q_nvvidconv = NULL, *q_nvosd = NULL,
             *tiler = NULL, *q_tiler = NULL, *sgie = NULL, *fps_sink = NULL,
             *q_sgie = NULL, *dsexample, *q_dsexample, *q_pgie;

  GOptionContext *ctx = NULL;
  GOptionGroup *group = NULL;
  GError *error = NULL;

  GstElement *msgconv = NULL, *msgbroker = NULL, *tee = NULL;
  GstElement *queueConv = NULL, *queueBroker = NULL;
  GstPad *osd_sink_pad = NULL;
  GstPad *tee_render_pad = NULL;
  GstPad *tee_msg_pad = NULL;
  GstPad *sink_pad = NULL;
  GstPad *src_pad = NULL;
  GstPad *queue_src_pad = NULL;
  GstPad *tiler_sink_pad = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  guint i;

  gchar *fps_msg;
  guint delay_show_FPS = 0;

  ctx = g_option_context_new("Nvidia DeepStream Test4");
  group = g_option_group_new("test4", NULL, NULL, NULL, NULL);
  g_option_group_add_entries(group, entries);

  if (!g_option_context_parse(ctx, &num_sources, &sources, &error)) {
    g_option_context_free(ctx);
    g_printerr("%s", error->message);
    return -1;
  }
  g_option_context_free(ctx);

  int current_device = 0;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  gst_init(&num_sources, &sources);
  loop = g_main_loop_new(NULL, FALSE);
  pipeline = gst_pipeline_new("FaceRecognition");
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
  g_object_set(G_OBJECT(streammux), "batch-size", num_sources, NULL);
  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
               MUXER_OUTPUT_HEIGHT, "batched-push-timeout",
               MUXER_BATCH_TIMEOUT_USEC, "enable-padding", TRUE, "live-source",
               0, "nvbuf-memory-type", 3, "gpu-id", 0, NULL);

  dsexample = gst_element_factory_make("dsexample", "example-plugin");
  g_object_set(G_OBJECT(dsexample), "full-frame", FALSE, "blur-objects", TRUE,
               "unique-id", 15, "gpu-id", 0, NULL);

  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
  g_object_set(G_OBJECT(nvvidconv), "nvbuf-memory-type", 3, NULL);

  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
  g_object_set(G_OBJECT(nvosd), "process-mode", OSD_PROCESS_MODE,
               "display-text", OSD_DISPLAY_TEXT, NULL);
  sgie = gst_element_factory_make("nvinfer", "face_extraction");
  g_object_set(G_OBJECT(sgie), "config-file-path", SGIE_CONFIG_FILE,
               "process-mode", 2, "unique-id", 2, NULL);

  sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
  g_object_set(G_OBJECT(sink), "qos", 0, NULL);
  g_object_set(G_OBJECT(sink), "sync", FALSE, "nvbuf-memory-type", 3, NULL);
  tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
  g_object_set(G_OBJECT(tiler), "rows", 1, "columns",
               (guint)ceil(1.0 * num_sources / 4), "width", MUXER_OUTPUT_WIDTH,
               "height", MUXER_OUTPUT_HEIGHT, "nvbuf-memory-type", 3, NULL);

  /////////////////////////////////////////////////define algorithm

  nvtracker = gst_element_factory_make("nvtracker", "tracker");
  if (!set_tracker_properties(nvtracker)) {
    g_printerr("Failed to set tracker properties. Exiting.\n");
    return -1;
  }
  pgie = gst_element_factory_make("nvinfer", "face_detection");
  g_object_set(G_OBJECT(pgie), "batch-size", num_sources, "config-file-path",
               PGIE_CONFIG_FILE, "unique-id", 1, NULL);

  // sgie = gst_element_factory_make("nvinfer", "face_extraction");
  // g_object_set(G_OBJECT(sgie), "config-file-path", SGIE_CONFIG_FILE,
  //              "output-tensor-meta", TRUE, "process-mode", 2, NULL);

  ///////////////////////////////////////////////////define queue

  q_nvtracker = gst_element_factory_make("queue", "q_nvtracker");
  q_nvosd = gst_element_factory_make("queue", "q_nvosd");
  q_nvvidconv = gst_element_factory_make("queue", "q_nvvidconv");
  q_tiler = gst_element_factory_make("queue", "q_tiler");
  q_sgie = gst_element_factory_make("queue", "q_sgie");
  q_pgie = gst_element_factory_make("queue", "q_pgie");
  q_dsexample = gst_element_factory_make("queue", "q_dsexample");
  fps_sink = gst_element_factory_make("fpsdisplaysink", "fps-display");
  g_object_set(G_OBJECT(fps_sink), "text-overlay", FALSE, "video-sink", sink,
               "sync", FALSE, NULL);

  g_signal_connect(fps_sink, "fps-measurements",
                   G_CALLBACK(fps_measurements_callback), NULL);

  ////////////////////////////////////message///////////////////////////

  msgconv = gst_element_factory_make("nvmsgconv", "nvmsg-converter");
  /* Create msg broker to send payload to server */
  msgbroker = gst_element_factory_make("nvmsgbroker", "nvmsg-broker");
  /* Create tee to render buffer and send message simultaneously*/
  tee = gst_element_factory_make("tee", "nvsink-tee");
  queueConv = gst_element_factory_make("queue", "nvtee-que1");
  queueBroker = gst_element_factory_make("queue", "nvtee-que2");
  g_object_set(G_OBJECT(msgconv), "config", MSCONV_CONFIG_FILE, NULL);
  g_object_set(G_OBJECT(msgconv), "payload-type", schema_type, NULL);
  g_object_set(G_OBJECT(msgbroker), "proto-lib", proto_lib, "conn-str",
               conn_str, "sync", FALSE, NULL);

  if (topic) {
    g_object_set(G_OBJECT(msgbroker), "topic", topic, NULL);
  }

  if (cfg_file) {
    g_object_set(G_OBJECT(msgbroker), "config", cfg_file, NULL);
  }

  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  gst_bin_add_many(GST_BIN(pipeline), q_pgie, pgie, q_nvtracker, nvtracker,
                   q_sgie, sgie, q_nvvidconv, nvvidconv, q_tiler, tiler,
                   q_dsexample, dsexample, q_nvosd, nvosd, sink, fps_sink, tee,
                   queueBroker, queueConv, msgconv, msgbroker, NULL);

  ///////////////////////////////////getsource///////////////////////////
  gst_bin_add(GST_BIN(pipeline), streammux);

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = {};
    GstElement *source_bin = create_source_bin(i, sources[i + 1]);

    if (!source_bin) {
      g_printerr("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add(GST_BIN(pipeline), source_bin);

    g_snprintf(pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad(streammux, pad_name);
    if (!sinkpad) {
      g_printerr("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad(source_bin, "src");
    if (!srcpad) {
      g_printerr("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
  }

  //////////////////////////////////////////////////////////////////////////////////////
  gst_element_link_many(streammux, pgie, q_nvvidconv, nvvidconv, q_nvtracker,
                        nvtracker, q_dsexample, dsexample, q_sgie, sgie,
                        q_tiler, tiler, q_nvosd, nvosd, tee,
                        NULL); //
  if (!gst_element_link_many(queueConv, msgconv, msgbroker, NULL)) {
    g_printerr("Elements could not be linked. Exiting.\n");
    return -1;
  }
  if (!gst_element_link(queueBroker, fps_sink)) {
    g_printerr("Elements could not be linked. Exiting.\n");
    return -1;
  }

  // add kafka
  sink_pad = gst_element_get_static_pad(queueConv, "sink");
  tee_msg_pad = gst_element_get_request_pad(tee, "src_%u");
  tee_render_pad = gst_element_get_request_pad(tee, "src_%u");
  if (!tee_msg_pad || !tee_render_pad) {
    g_printerr("Unable to get request pads\n");
    return -1;
  }

  if (gst_pad_link(tee_msg_pad, sink_pad) != GST_PAD_LINK_OK) {
    g_printerr("Unable to link tee and message converter\n");
    gst_object_unref(sink_pad);
    return -1;
  }

  gst_object_unref(sink_pad);

  sink_pad = gst_element_get_static_pad(queueBroker, "sink");
  if (gst_pad_link(tee_render_pad, sink_pad) != GST_PAD_LINK_OK) {
    g_printerr("Unable to link tee and render\n");
    gst_object_unref(sink_pad);
    return -1;
  }

  gst_object_unref(sink_pad);
  /////////////////////////////////// get output froms gie

  // tiler_sink_pad = gst_element_get_static_pad(nvtracker, "src");
  // gst_pad_add_probe(tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
  //                   pgie_pad_buffer_probe, NULL, NULL);

  ////////////////////// post process the output from pgie

  queue_src_pad = gst_element_get_static_pad(q_tiler, "src");
  gst_pad_add_probe(queue_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                    tiler_src_pad_buffer_probe, (gpointer)fps_sink, NULL);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

  return 0;
}