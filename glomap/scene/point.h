#pragma once

#include "glomap/types.h"

namespace glomap {
struct Point {
  Eigen::Vector3d position;
  Eigen::Vector3d normal;
  Point()
      : position(Eigen::Vector3d::Zero()), normal(Eigen::Vector3d::Zero()) {}
  Point(const Eigen::Vector3d& pos, const Eigen::Vector3d& norm)
      : position(pos), normal(norm) {}
};

struct PointCloudAdaptor {
  const std::vector<Point>& points;

  PointCloudAdaptor(const std::vector<Point>& pts) : points(pts) {}

  inline size_t kdtree_get_point_count() const { return points.size(); }

  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return points[idx].position[dim];
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }
};
}  // namespace glomap