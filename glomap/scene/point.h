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
}  // namespace glomap