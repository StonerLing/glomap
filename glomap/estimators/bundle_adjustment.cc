#include "bundle_adjustment.h"

#include "glomap/io/tinyply.h"
#include "glomap/processors/reconstruction_aligner.h"
#include "glomap/processors/reconstruction_normalizer.h"

#include <colmap/estimators/cost_functions.h>
#include <colmap/estimators/manifold.h>
#include <colmap/sensor/models.h>
#include <colmap/util/cuda.h>
#include <colmap/util/misc.h>

namespace glomap {
namespace {
// Extracts valid pose priors from a collection of images.
std::unordered_map<image_t, colmap::PosePrior>
ExtractsValidPosePriorsFromImgaes(
    const std::unordered_map<image_t, Image>& images) {
  std::unordered_map<image_t, colmap::PosePrior> pose_priors;
  for (const auto& [image_t, image] : images) {
    if (image.pose_prior.has_value()) {
      pose_priors[image_t] = image.pose_prior.value();
    }
  }
  return pose_priors;
}

// Denormalizes the reconstruction from the normalized coordinate system back to
// the metric space.
void DenormalizeReconstruction(const colmap::Sim3d& normalized_from_metric,
                               std::unordered_map<image_t, Image>& images,
                               std::unordered_map<track_t, Track>& tracks) {
  colmap::Sim3d tform = Inverse(normalized_from_metric);
  for (auto& [_, image] : images) {
    if (image.is_registered) {
      image.cam_from_world = TransformCameraWorld(tform, image.cam_from_world);
    }
  }

  for (auto& [_, track] : tracks) {
    track.xyz = tform * track.xyz;
  }
}
}  // namespace

bool BundleAdjuster::Solve(const ViewGraph& view_graph,
                           std::unordered_map<camera_t, Camera>& cameras,
                           std::unordered_map<image_t, Image>& images,
                           std::unordered_map<track_t, Track>& tracks) {
  // Check if the input data is valid
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }
  if (tracks.empty()) {
    LOG(ERROR) << "Number of tracks = " << tracks.size();
    return false;
  }

  // Reset the problem
  Reset();

  // Add the constraints that the point tracks impose on the problem
  AddPointToCameraConstraints(view_graph, cameras, images, tracks);

  // Add the cameras and points to the parameter groups for schur-based
  // optimization
  AddCamerasAndPointsToParameterGroups(cameras, images, tracks);

  // Parameterize the variables
  ParameterizeVariables(cameras, images, tracks);

  // Set the solver options.
  ceres::Solver::Summary summary;

  int num_images = images.size();
#ifdef GLOMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (options_.use_gpu && num_images >= options_.min_num_images_gpu_solver) {
    cuda_solver_enabled = true;
    options_.solver_options.dense_linear_algebra_library_type = ceres::CUDA;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without CUDA support. Falling back to CPU-based dense "
           "solvers.";
  }
#endif

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3)) && \
    !defined(CERES_NO_CUDSS)
  if (options_.use_gpu && num_images >= options_.min_num_images_gpu_solver) {
    cuda_solver_enabled = true;
    options_.solver_options.sparse_linear_algebra_library_type =
        ceres::CUDA_SPARSE;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without cuDSS support. Falling back to CPU-based sparse "
           "solvers.";
  }
#endif

  if (cuda_solver_enabled) {
    const std::vector<int> gpu_indices =
        colmap::CSVToVector<int>(options_.gpu_index);
    THROW_CHECK_GT(gpu_indices.size(), 0);
    colmap::SetBestCudaDevice(gpu_indices[0]);
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but COLMAP was "
           "compiled without CUDA support. Falling back to CPU-based "
           "solvers.";
  }
#endif  // GLOMAP_CUDA_ENABLED

  // Do not use the iterative solver, as it does not seem to be helpful
  options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;

  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);
  if (VLOG_IS_ON(2))
    LOG(INFO) << summary.FullReport();
  else
    LOG(INFO) << summary.BriefReport();

  return summary.IsSolutionUsable();
}

void BundleAdjuster::Reset() {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();
}

void BundleAdjuster::AddPointToCameraConstraints(
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  for (auto& [track_id, track] : tracks) {
    if (track.observations.size() < options_.min_num_view_per_track) continue;

    for (const auto& observation : tracks[track_id].observations) {
      if (images.find(observation.first) == images.end()) continue;

      Image& image = images[observation.first];

      ceres::CostFunction* cost_function =
          colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
              cameras[image.camera_id].model_id,
              image.features[observation.second]);

      if (cost_function != nullptr) {
        problem_->AddResidualBlock(
            cost_function,
            loss_function_.get(),
            image.cam_from_world.rotation.coeffs().data(),
            image.cam_from_world.translation.data(),
            tracks[track_id].xyz.data(),
            cameras[image.camera_id].params.data());
      } else {
        LOG(ERROR) << "Camera model not supported: "
                   << colmap::CameraModelIdToName(
                          cameras[image.camera_id].model_id);
      }
    }
  }
}

void BundleAdjuster::AddCamerasAndPointsToParameterGroups(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  if (tracks.size() == 0) return;

  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();
  // Add point parameters to group 0.
  for (auto& [track_id, track] : tracks) {
    if (problem_->HasParameterBlock(track.xyz.data()))
      parameter_ordering->AddElementToGroup(track.xyz.data(), 0);
  }

  // Add camera parameters to group 1.
  for (auto& [image_id, image] : images) {
    if (problem_->HasParameterBlock(image.cam_from_world.translation.data())) {
      parameter_ordering->AddElementToGroup(
          image.cam_from_world.translation.data(), 1);
      parameter_ordering->AddElementToGroup(
          image.cam_from_world.rotation.coeffs().data(), 1);
    }
  }

  // Add camera parameters to group 1.
  for (auto& [camera_id, camera] : cameras) {
    if (problem_->HasParameterBlock(camera.params.data()))
      parameter_ordering->AddElementToGroup(camera.params.data(), 1);
  }
}

void BundleAdjuster::ParameterizeVariables(
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  image_t center;

  // Parameterize rotations, and set rotations and translations to be constant
  // if desired FUTURE: Consider fix the scale of the reconstruction
  int counter = 0;
  for (auto& [image_id, image] : images) {
    if (problem_->HasParameterBlock(
            image.cam_from_world.rotation.coeffs().data())) {
      colmap::SetQuaternionManifold(
          problem_.get(), image.cam_from_world.rotation.coeffs().data());

      if (counter == 0) {
        center = image_id;
        counter++;
      }
      if (!options_.optimize_rotations)
        problem_->SetParameterBlockConstant(
            image.cam_from_world.rotation.coeffs().data());
      if (!options_.optimize_translation)
        problem_->SetParameterBlockConstant(
            image.cam_from_world.translation.data());
    }
  }

  if (counter > 0) {
    // Set the first camera to be fixed to remove the gauge ambiguity.
    problem_->SetParameterBlockConstant(
        images[center].cam_from_world.rotation.coeffs().data());
    problem_->SetParameterBlockConstant(
        images[center].cam_from_world.translation.data());
  }

  // Parameterize the camera parameters, or set them to be constant if desired
  if (options_.optimize_intrinsics && !options_.optimize_principal_point) {
    for (auto& [camera_id, camera] : cameras) {
      if (problem_->HasParameterBlock(camera.params.data())) {
        std::vector<int> principal_point_idxs;
        for (auto idx : camera.PrincipalPointIdxs()) {
          principal_point_idxs.push_back(idx);
        }
        colmap::SetSubsetManifold(camera.params.size(),
                                  principal_point_idxs,
                                  problem_.get(),
                                  camera.params.data());
      }
    }
  } else if (!options_.optimize_intrinsics &&
             !options_.optimize_principal_point) {
    for (auto& [camera_id, camera] : cameras) {
      if (problem_->HasParameterBlock(camera.params.data())) {
        problem_->SetParameterBlockConstant(camera.params.data());
      }
    }
  }

  if (!options_.optimize_points) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data())) {
        problem_->SetParameterBlockConstant(track.xyz.data());
      }
    }
  }
}

PosePriorBundleAdjuster::PosePriorBundleAdjuster(
    const BundleAdjusterOptions& options,
    const PosePriorBundleAdjusterOptions& prior_options)
    : BundleAdjuster(options), prior_options_(prior_options) {
  if (prior_options_.use_robust_loss_on_prior_position) {
    prior_options_.prior_position_loss_function =
        std::make_shared<ceres::ScaledLoss>(
            new ceres::CauchyLoss(prior_options_.prior_position_loss_threshold),
            prior_options_.prior_position_scaled_loss_factor,
            ceres::DO_NOT_TAKE_OWNERSHIP);
  } else {
    // nullptr for loss function means identity loss.
    prior_options_.prior_position_loss_function =
        std::make_shared<ceres::ScaledLoss>(
            nullptr,
            prior_options_.prior_position_scaled_loss_factor,
            ceres::DO_NOT_TAKE_OWNERSHIP);
  }
}

bool PosePriorBundleAdjuster::Solve(
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // Check if the input data is valid
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }
  if (tracks.empty()) {
    LOG(ERROR) << "Number of tracks = " << tracks.size();
    return false;
  }

  // Extracts all valid pose priors from images
  pose_priors_ = ExtractsValidPosePriorsFromImgaes(images);

  // Reset the problem
  Reset();

  // Try to align the reconstruction to pose position prior
  const bool use_prior_position =
      AlignReconstruction(pose_priors_, images, tracks);

  // Add the constraints that the point tracks impose on the problem
  AddPointToCameraConstraints(view_graph, cameras, images, tracks);

  Sim3d normalized_from_metric;
  if (use_prior_position) {
    normalized_from_metric =
        NormalizeReconstruction(cameras, images, tracks, true);

    AddPosePositionPriorConstraints(
        pose_priors_, normalized_from_metric, images);
  }

  // Add the cameras and points to the parameter groups for schur-based
  // optimization
  AddCamerasAndPointsToParameterGroups(cameras, images, tracks);

  // Parameterize the variables
  ParameterizeVariables(cameras, images, tracks);

  // If we confirm to optimize translation with prior position, than make sure
  // all translations are variable.
  if (use_prior_position && options_.optimize_translation) {
    std::for_each(images.begin(), images.end(), [&](auto& id_image_pair) {
      problem_->SetParameterBlockVariable(
          id_image_pair.second.cam_from_world.translation.data());
    });
    ;
  }

  // Set the solver options.
  ceres::Solver::Summary summary;

  // Do not use the iterative solver, as it does not seem to be helpful
  options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;

  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);
  if (VLOG_IS_ON(2))
    LOG(INFO) << summary.FullReport();
  else
    LOG(INFO) << summary.BriefReport();

  // Transform the reconstruction back.
  if (use_prior_position) {
    DenormalizeReconstruction(normalized_from_metric, images, tracks);
  }
  return summary.IsSolutionUsable();
}

bool PosePriorBundleAdjuster::AlignReconstruction(
    const std::unordered_map<image_t, colmap::PosePrior>& pose_priors,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  double max_error = -1;
  if (prior_options_.ransac_max_error > 0) {
    max_error = prior_options_.ransac_max_error;
  } else {
    double max_stddev_sum = 0;
    size_t num_valid_covs = 0;
    for (const auto& [_, pose_prior] : pose_priors) {
      if (pose_prior.IsCovarianceValid()) {
        const double max_stddev =
            std::sqrt(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(
                          pose_prior.position_covariance)
                          .eigenvalues()
                          .maxCoeff());
        max_stddev_sum += max_stddev;
        ++num_valid_covs;
      }
    }
    if (num_valid_covs == 0) {
      LOG(WARNING) << "No pose priors with valid covariance found.";
      return false;
    }
    // Set max error at the 3 sigma confidence interval. Assumes no outliers.
    max_error = 3 * max_stddev_sum / num_valid_covs;
  }

  return AlignReconstructionToPosePositionPriors(
      pose_priors, images, tracks, max_error);
  ;
}

void PosePriorBundleAdjuster::AddPosePositionPriorConstraints(
    const std::unordered_map<image_t, colmap::PosePrior>& pose_priors,
    const Sim3d& normalized_from_metric,
    std::unordered_map<image_t, Image>& images) {
  const int num_images = images.size();

  // Add pose position prior constraints
  for (auto& [image_id, image] : images) {
    const colmap::PosePrior& pose_prior = pose_priors.at(image_id);
    ceres::CostFunction* cost_function = colmap::CovarianceWeightedCostFunctor<
        colmap::AbsolutePosePositionPriorCostFunctor>::
        Create(pose_prior.position_covariance,
               normalized_from_metric * pose_prior.position);

    if (cost_function != nullptr) {
      problem_->AddResidualBlock(
          cost_function,
          prior_options_.prior_position_loss_function.get(),
          image.cam_from_world.rotation.coeffs().data(),
          image.cam_from_world.translation.data());
    } else {
      LOG(ERROR) << "Could not create position prior cost function for image: "
                 << image_id;
    }
  }
}

LidarPriorBundleAdjuster::LidarPriorBundleAdjuster(
    const BundleAdjusterOptions& options,
    const LidarPriorBundleAdjusterOptions& lidar_options)
    : BundleAdjuster(options), lidar_options_(lidar_options) {
  if (lidar_options_.use_robust_loss_on_prior_position) {
    lidar_options_.prior_lidar_loss_function =
        std::make_shared<ceres::ScaledLoss>(
            new ceres::CauchyLoss(lidar_options_.prior_position_loss_threshold),
            lidar_options_.prior_position_scaled_loss_factor,
            ceres::DO_NOT_TAKE_OWNERSHIP);
  } else {
    // nullptr for loss function means identity loss.
    lidar_options_.prior_lidar_loss_function =
        std::make_shared<ceres::ScaledLoss>(
            nullptr,
            lidar_options_.prior_position_scaled_loss_factor,
            ceres::DO_NOT_TAKE_OWNERSHIP);
  }
}

bool LidarPriorBundleAdjuster::SolveLidar(
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::vector<Point> points3D,
    std::unordered_map<track_t, Track>& tracks,
    double weight) {
  // Check if the input data is valid
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }
  if (tracks.empty()) {
    LOG(ERROR) << "Number of tracks = " << tracks.size();
    return false;
  }

  // Reset the problem
  Reset();

  // Try to align the reconstruction to pose position prior
  constexpr bool use_prior_lidar = true;

  // Add the constraints that the point tracks impose on the problem
  AddPointToCameraConstraints(view_graph, cameras, images, tracks);

  Sim3d normalized_from_metric;
  if (use_prior_lidar) {
    normalized_from_metric =
        NormalizeReconstruction(cameras, images, tracks, false);

    AddLidarPositionPriorConstraints(
        points3D, normalized_from_metric, tracks, weight);
  }

  // Add the cameras and points to the parameter groups for schur-based
  // optimization
  AddCamerasAndPointsToParameterGroups(cameras, images, tracks);

  // Parameterize the variables
  ParameterizeVariables(cameras, images, tracks);

  // If we confirm to optimize with LIDAR prior, make sure all track points
  // are variable
  if (use_prior_lidar && options_.optimize_points) {
    std::for_each(tracks.begin(), tracks.end(), [&](auto& id_track_pair) {
      if (problem_->HasParameterBlock(id_track_pair.second.xyz.data())) {
        problem_->SetParameterBlockVariable(id_track_pair.second.xyz.data());
      }
    });
  }

  // Set the solver options.
  ceres::Solver::Summary summary;

  // Do not use the iterative solver, as it does not seem to be helpful
  options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;

  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);
  if (VLOG_IS_ON(2))
    LOG(INFO) << summary.FullReport();
  else
    LOG(INFO) << summary.BriefReport();

  // Transform the reconstruction back.
  if (use_prior_lidar) {
    DenormalizeReconstruction(normalized_from_metric, images, tracks);
  }
  return summary.IsSolutionUsable();
}

bool LidarPriorBundleAdjuster::AlignReconstruction(
    const std::unordered_map<image_t, colmap::PosePrior>& pose_priors,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  double max_error = -1;
  if (lidar_options_.ransac_max_error > 0) {
    max_error = lidar_options_.ransac_max_error;
  } else {
    double max_stddev_sum = 0;
    size_t num_valid_covs = 0;
    for (const auto& [_, pose_prior] : pose_priors) {
      if (pose_prior.IsCovarianceValid()) {
        const double max_stddev =
            std::sqrt(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(
                          pose_prior.position_covariance)
                          .eigenvalues()
                          .maxCoeff());
        max_stddev_sum += max_stddev;
        ++num_valid_covs;
      }
    }
    if (num_valid_covs == 0) {
      LOG(WARNING) << "No pose priors with valid covariance found.";
      return false;
    }
    // Set max error at the 3 sigma confidence interval. Assumes no outliers.
    max_error = 3 * max_stddev_sum / num_valid_covs;
  }

  return AlignReconstructionToPosePositionPriors(
      pose_priors, images, tracks, max_error);
  ;
}

// Point-to-point cost functor for LIDAR constraints
struct PointToPointCostFunctor {
  PointToPointCostFunctor(const Eigen::Vector3d& lidar_point,
                          double weight = 1.0)
      : lidar_point_(lidar_point), weight_(weight) {}

  template <typename T>
  bool operator()(const T* const track_point, T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residuals);
    residual =
        weight_ * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(track_point) -
                   lidar_point_.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& lidar_point,
                                     double weight = 1.0) {
    return new ceres::AutoDiffCostFunction<PointToPointCostFunctor, 3, 3>(
        new PointToPointCostFunctor(lidar_point, weight));
  }

  Eigen::Vector3d lidar_point_;
  double weight_;
};

// Point-to-plane cost functor for LIDAR constraints
struct PointToPlaneCostFunctor {
  PointToPlaneCostFunctor(const Eigen::Vector3d& lidar_point,
                          const Eigen::Vector3d& lidar_normal,
                          double weight = 1.0)
      : lidar_point_(lidar_point),
        lidar_normal_(lidar_normal),
        weight_(weight) {}

  template <typename T>
  bool operator()(const T* const track_point, T* residuals) const {
    Eigen::Matrix<T, 3, 1> point_diff =
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(track_point) -
        lidar_point_.cast<T>();
    residuals[0] = weight_ * point_diff.dot(lidar_normal_.cast<T>());
    // weight_ * ceres::abs(point_diff.dot(lidar_normal_.cast<T>()));
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& lidar_point,
                                     const Eigen::Vector3d& lidar_normal,
                                     double weight = 1.0) {
    return new ceres::AutoDiffCostFunction<PointToPlaneCostFunctor, 1, 3>(
        new PointToPlaneCostFunctor(lidar_point, lidar_normal, weight));
  }

  Eigen::Vector3d lidar_point_;
  Eigen::Vector3d lidar_normal_;
  double weight_;
};

void SavePointsToPLY(const std::vector<Eigen::Vector3d>& points,
                     const std::string& filename) {
  std::filebuf fb;
  fb.open(filename, std::ios::out | std::ios::binary);
  std::ostream os(&fb);

  tinyply::PlyFile ply_file;

  std::vector<float> vertices;
  vertices.reserve(points.size() * 3);
  for (const auto& point : points) {
    vertices.push_back(static_cast<float>(point.x()));
    vertices.push_back(static_cast<float>(point.y()));
    vertices.push_back(static_cast<float>(point.z()));
  }

  ply_file.add_properties_to_element(
      "vertex",
      {"x", "y", "z"},
      tinyply::Type::FLOAT32,
      vertices.size() / 3,
      reinterpret_cast<uint8_t*>(vertices.data()),
      tinyply::Type::INVALID,
      0);

  ply_file.write(os, true);

  fb.close();
}

void SavePointsToPLY(const std::vector<Point>& points,
                     const std::string& filename) {
  std::filebuf fb;
  fb.open(filename, std::ios::out | std::ios::binary);
  std::ostream os(&fb);

  tinyply::PlyFile ply_file;

  std::vector<float> vertices;
  std::vector<float> normals;
  vertices.reserve(points.size() * 3);
  normals.reserve(points.size() * 3);

  for (const auto& point : points) {
    vertices.push_back(static_cast<float>(point.position.x()));
    vertices.push_back(static_cast<float>(point.position.y()));
    vertices.push_back(static_cast<float>(point.position.z()));

    normals.push_back(static_cast<float>(point.normal.x()));
    normals.push_back(static_cast<float>(point.normal.y()));
    normals.push_back(static_cast<float>(point.normal.z()));
  }

  ply_file.add_properties_to_element(
      "vertex",
      {"x", "y", "z"},
      tinyply::Type::FLOAT32,
      vertices.size() / 3,
      reinterpret_cast<uint8_t*>(vertices.data()),
      tinyply::Type::INVALID,
      0);

  ply_file.add_properties_to_element("vertex",
                                     {"nx", "ny", "nz"},
                                     tinyply::Type::FLOAT32,
                                     normals.size() / 3,
                                     reinterpret_cast<uint8_t*>(normals.data()),
                                     tinyply::Type::INVALID,
                                     0);

  // 写入文件
  ply_file.write(os, true);

  fb.close();
}

void LidarPriorBundleAdjuster::AddLidarPositionPriorConstraints(
    const std::vector<Point> points3D,
    const Sim3d& normalized_from_metric,
    std::unordered_map<track_t, Track>& tracks,
    double weight) {
  const int num_tracks = tracks.size();

  if (points3D.empty()) {
    LOG(WARNING) << "No 3D points provided for LIDAR prior constraints";
    return;
  }

  // Create a point cloud adaptor for nanoflann
  std::vector<Point> points3D_transformed;
  points3D_transformed.reserve(points3D.size());
  for (const auto& point : points3D) {
    Point transformed_point;
    transformed_point.position = normalized_from_metric * point.position;
    transformed_point.normal = normalized_from_metric.rotation * point.normal;
    points3D_transformed.push_back(transformed_point);
  }

  // Create point cloud adaptor instance
  PointCloudAdaptor pointCloudAdaptor(points3D_transformed);

  // Create KD-tree for 3D points
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
      PointCloudAdaptor,
      3>;

  KDTree kdtree(
      3, pointCloudAdaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  kdtree.buildIndex();

  std::vector<Eigen::Vector3d> points3D_src;
  std::vector<Point> points3D_dst;
  points3D_src.reserve(tracks.size());
  points3D_dst.reserve(tracks.size());

  for (auto& [track_id, track] : tracks) {
    // Query point (the track's 3D position)
    const Eigen::Vector3d query_point = track.xyz;
    // Find the nearest neighbor
    size_t nearest_index;
    double nearest_distance;
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&nearest_index, &nearest_distance);
    kdtree.findNeighbors(resultSet, query_point.data());

    // Add constraint using the nearest LIDAR point
    const Point& nearest_point = points3D_transformed[nearest_index];
    points3D_dst.emplace_back(nearest_point);
    points3D_src.emplace_back(track.xyz);
    // Create cost function using point-to-plane distance as constraint
    ceres::CostFunction* cost_function = PointToPlaneCostFunctor::Create(
        nearest_point.position, nearest_point.normal, weight);

    // ceres::CostFunction* cost_function =
    //     PointToPointCostFunctor::Create(nearest_point.position, weight);

    if (cost_function != nullptr) {
      problem_->AddResidualBlock(cost_function,
                                 lidar_options_.prior_lidar_loss_function.get(),
                                 track.xyz.data());
    } else {
      LOG(ERROR)
          << "Could not create LIDAR position prior cost function for track: "
          << track_id;
    }
  }
  // std::string output_path =
  //     "C:/Users/Admin/Desktop/Data/HP300/10-17/output/undistorted_project/"
  //     "sparse_icp/0";
  // SavePointsToPLY(points3D_src, output_path + "/points3D_src.ply");
  // SavePointsToPLY(points3D_dst, output_path + "/points3D_dst.ply");
}
}  // namespace glomap
