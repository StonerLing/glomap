#include "glomap/controllers/global_mapper.h"

#include "glomap/controllers/global_mapper_lidar.h"
#include "glomap/controllers/option_manager.h"
#include "glomap/io/colmap_io.h"
#include "glomap/io/tinyply.h"
#include "glomap/types.h"

#include <colmap/scene/reconstruction.h>
#include <colmap/util/endian.h>
#include <colmap/util/file.h>
#include <colmap/util/misc.h>
#include <colmap/util/timer.h>

namespace glomap {
namespace {

std::vector<Point> ReadPoints3D(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(stream, path);

  std::vector<Point> points3D;

  tinyply::PlyFile file;
  file.parse_header(stream);

  LOG(INFO) << "PLY head info:";
  for (auto& comment : file.get_comments()) {
    LOG(INFO) << "comment: " << comment;
  }
  for (auto& element : file.get_elements()) {
    LOG(INFO) << "element: " << element.name << " (" << element.size << ")";
    for (auto& property : element.properties) {
      LOG(INFO) << "  property: " << property.name;
    }
  }

  std::shared_ptr<tinyply::PlyData> vertices, normals;

  try {
    vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to request vertex properties: " << e.what();
  }

  try {
    normals =
        file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
  } catch (const std::exception& e) {
    LOG(INFO) << "Normals not available in PLY file: " << e.what();
  }

  // Read the data
  file.read(stream);

  if (!vertices) {
    LOG(ERROR) << "No vertex data found in PLY file";
    return points3D;
  }

  // Check data type
  if (vertices->t != tinyply::Type::FLOAT32 &&
      vertices->t != tinyply::Type::FLOAT64) {
    LOG(ERROR) << "Unsupported vertex data type";
    return points3D;
  }

  points3D.reserve(vertices->count);

  const size_t vertexCount = vertices->count;
  const uint8_t* vertexData = vertices->buffer.get_const();

  // Handle normals if available
  const uint8_t* normalData = nullptr;
  size_t normalCount = 0;
  tinyply::Type normalType = tinyply::Type::INVALID;

  if (normals) {
    normalData = normals->buffer.get_const();
    normalCount = normals->count;
    normalType = normals->t;

    if (normalCount != vertexCount) {
      LOG(WARNING)
          << "Normal count does not match vertex count, ignoring normals";
      normalData = nullptr;
    }

    if (normalType != tinyply::Type::FLOAT32 &&
        normalType != tinyply::Type::FLOAT64) {
      LOG(WARNING) << "Unsupported normal data type, ignoring normals";
      normalData = nullptr;
    }
  }

  const size_t stride =
      (vertices->t == tinyply::Type::FLOAT32) ? sizeof(float) : sizeof(double);
  const size_t normalStride =
      normalData ? ((normalType == tinyply::Type::FLOAT32) ? sizeof(float)
                                                           : sizeof(double))
                 : 0;

  for (size_t i = 0; i < vertexCount; ++i) {
    Eigen::Vector3d position, normal = Eigen::Vector3d::Zero();

    if (vertices->t == tinyply::Type::FLOAT32) {
      const float* data =
          reinterpret_cast<const float*>(vertexData + i * 3 * stride);
      position = Eigen::Vector3d(static_cast<double>(data[0]),
                                 static_cast<double>(data[1]),
                                 static_cast<double>(data[2]));
    } else {  // FLOAT64
      const double* data =
          reinterpret_cast<const double*>(vertexData + i * 3 * stride);
      position = Eigen::Vector3d(data[0], data[1], data[2]);
    }

    if (normalData) {
      if (normalType == tinyply::Type::FLOAT32) {
        const float* data =
            reinterpret_cast<const float*>(normalData + i * 3 * normalStride);
        normal = Eigen::Vector3d(static_cast<double>(data[0]),
                                 static_cast<double>(data[1]),
                                 static_cast<double>(data[2]));
      } else {  // FLOAT64
        const double* data =
            reinterpret_cast<const double*>(normalData + i * 3 * normalStride);
        normal = Eigen::Vector3d(data[0], data[1], data[2]);
      }
    }

    points3D.emplace_back(Point(position, normal));
  }

  LOG(INFO) << "Loaded " << points3D.size() << " points from PLY file";
  return points3D;
}

void UpdateDatabasePosePriorsCovariance(colmap::Database& database,
                                        const Eigen::Matrix3d& covariance) {
  colmap::DatabaseTransaction database_transaction(&database);

  LOG(INFO)
      << "Setting up database pose priors with the same covariance matrix: \n"
      << covariance << "\n";

  for (const auto& image : database.ReadAllImages()) {
    if (database.ExistsPosePrior(image.ImageId())) {
      colmap::PosePrior prior = database.ReadPosePrior(image.ImageId());
      prior.position_covariance = covariance;
      database.UpdatePosePrior(image.ImageId(), prior);
    }
  }
}
}  // namespace

// -------------------------------------
// Mappers starting from COLMAP database
// -------------------------------------
int RunMapper(int argc, char** argv) {
  std::string database_path;
  std::string output_path;

  std::string image_path = "";
  std::string constraint_type = "ONLY_POINTS";
  std::string output_format = "bin";

  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("constraint_type",
                           &constraint_type,
                           "{ONLY_POINTS, ONLY_CAMERAS, "
                           "POINTS_AND_CAMERAS_BALANCED, POINTS_AND_CAMERAS}");
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddGlobalMapperFullOptions();

  options.Parse(argc, argv);

  if (!colmap::ExistsFile(database_path)) {
    LOG(ERROR) << "`database_path` is not a file";
    return EXIT_FAILURE;
  }

  if (constraint_type == "ONLY_POINTS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_POINTS;
  } else if (constraint_type == "ONLY_CAMERAS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_CAMERAS;
  } else if (constraint_type == "POINTS_AND_CAMERAS_BALANCED") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED;
  } else if (constraint_type == "POINTS_AND_CAMERAS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS;
  } else {
    LOG(ERROR) << "Invalid constriant type";
    return EXIT_FAILURE;
  }

  // Check whether output_format is valid
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  // Load the database
  ViewGraph view_graph;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;

  colmap::Database database(database_path);
  ConvertDatabaseToGlomap(
      database,
      view_graph,
      cameras,
      images,
      options.mapper->opt_pose_prior.use_pose_position_prior);

  if (options.mapper->opt_pose_prior.overwrite_position_priors_covariance) {
    const Eigen::Matrix3d covariance =
        Eigen::Vector3d(options.mapper->opt_pose_prior.prior_position_std_x,
                        options.mapper->opt_pose_prior.prior_position_std_y,
                        options.mapper->opt_pose_prior.prior_position_std_z)
            .cwiseAbs2()
            .asDiagonal();
    UpdateDatabasePosePriorsCovariance(database, covariance);
  }

  if (view_graph.image_pairs.empty()) {
    LOG(ERROR) << "Can't continue without image pairs";
    return EXIT_FAILURE;
  }

  GlobalMapper global_mapper(*options.mapper);

  // Main solver
  LOG(INFO) << "Loaded database";
  colmap::Timer run_timer;
  run_timer.Start();
  global_mapper.Solve(database, view_graph, cameras, images, tracks);
  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteGlomapReconstruction(
      output_path, cameras, images, tracks, output_format, image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

// -------------------------------------
// Mappers starting from COLMAP reconstruction
// -------------------------------------
int RunMapperResume(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string image_path = "";
  std::string output_format = "bin";

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddGlobalMapperResumeFullOptions();

  options.Parse(argc, argv);

  if (!colmap::ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  // Check whether output_format is valid
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  // Load the reconstruction
  ViewGraph view_graph;       // dummy variable
  colmap::Database database;  // dummy variable

  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;
  colmap::Reconstruction reconstruction;
  reconstruction.Read(input_path);
  ConvertColmapToGlomap(reconstruction, cameras, images, tracks);

  GlobalMapper global_mapper(*options.mapper);

  // Main solver
  colmap::Timer run_timer;
  run_timer.Start();
  global_mapper.Solve(database, view_graph, cameras, images, tracks);
  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteGlomapReconstruction(
      output_path, cameras, images, tracks, output_format, image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

// -------------------------------------
// Mappers starting from COLMAP reconstruction and LiDAR
// -------------------------------------
int RunMapperLidar(int argc, char** argv) {
  std::string input_path;
  std::string lidar_path;
  std::string output_path;
  std::string image_path = "";
  std::string output_format = "bin";

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("lidar_path", &lidar_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddGlobalMapperResumeFullOptions();

  options.Parse(argc, argv);

  options.mapper->skip_global_positioning = true;

  if (!colmap::ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  // Check whether output_format is valid
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  // Load the reconstruction
  ViewGraph view_graph;       // dummy variable
  colmap::Database database;  // dummy variable

  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;
  colmap::Reconstruction reconstruction;
  reconstruction.Read(input_path);
  ConvertColmapToGlomap(reconstruction, cameras, images, tracks);

  std::vector<Point> points = ReadPoints3D(lidar_path);

  GlobalLidarMapper global_mapper(*options.mapper);

  // Main solver
  colmap::Timer run_timer;
  run_timer.Start();
  global_mapper.Solve(database, view_graph, cameras, images, points, tracks);
  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteGlomapReconstruction(
      output_path, cameras, images, tracks, output_format, image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

}  // namespace glomap
