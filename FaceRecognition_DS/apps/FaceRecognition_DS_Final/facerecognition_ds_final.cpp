// #include "FaceRecognition_DS_Final/alignment.h"
#include "FaceRecognition_DS_Final/facerecognition_ds_final.h"
#include "FaceRecognition_DS_Final/ExtractFeature.h"
#include "yaml-cpp/yaml.h"
#include <iostream>
#define config_file "../config.yaml"
char **sources;
int num_source;
void init_source(YAML::Node config)
{

  std::vector<std::string> cf_sources = config["Source"]["source"].as<std::vector<std::string>>();
  num_source = cf_sources.size();
  std::cout << num_source << " sources \n";
  sources = (char **)calloc(num_source + 1, sizeof(char *));
  for (int i = 1; i <= num_source; i++)
  {
    std::cout<<cf_sources[i-1]<<"\n";
    sources[i]=new char[cf_sources[i-1].length() + 1];
    strcpy(sources[i], cf_sources[i-1].c_str());
  }
}

int main(int argc, char *argv[])
{
  YAML::Node config = YAML::LoadFile(config_file);
  // std::cout<<config["RabbitMQ"]["conn_str"].as<std::string>();
  init_source(config);

  // cudaStream_t *stream = new cudaStream_t;
  // Logger *logger = new Logger;
  // IRuntime *runtime = createInferRuntime(*logger);
  // FaceExtraction *face_extractor = new FaceExtraction;
  // face_extractor->setup(
  //     stream, runtime, logger,
  //     "../apps/FaceRecognition_DS_Final/models/iresnet124_4.engine");

  // sources[2] = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4";
  // sources[1] = "file:///home/haobk/oneface.avi";

  // sources[1] = "rtsp://admin:=}ueVzm[r`@192.168.100.235:554";

  // sources[2] = "rtsp://admin:=}ueVzm[r`@192.168.100.159:554";
  // sources[1] = "file:///home/haobk/test.mp4";
  // sources[2] = "file:///home/haobk/hnh.mp4";
  // sources[1] = "rtsp://admin:CUSTRJ@192.168.100.28:554";
  // sources[4] = "file:///home/haobk/hnh.mp4";
  std::cout<<"sources "<<sources[1]<<"\n";
  FaceRecognition_DS_Final(num_source, sources, config);
  return 0;
}