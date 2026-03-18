#include "peterson_r_graham_scan_omp/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>
#include <vector>

#include "peterson_r_graham_scan_omp/common/include/common.hpp"

namespace peterson_r_graham_scan_omp {

namespace {
constexpr double kTolerance = 1e-12;

double CalculateOrientation(const Point2D &origin, const Point2D &a, const Point2D &b) {
  return ((a.coord_x - origin.coord_x) * (b.coord_y - origin.coord_y)) -
         ((a.coord_y - origin.coord_y) * (b.coord_x - origin.coord_x));
}

double CalculateSquaredDistance(const Point2D &first, const Point2D &second) {
  const double dx = first.coord_x - second.coord_x;
  const double dy = first.coord_y - second.coord_y;
  return (dx * dx) + (dy * dy);
}

class PointComparator {
 public:
  explicit PointComparator(const Point2D *reference) : origin_ptr_(reference) {}

  bool operator()(const Point2D &lhs, const Point2D &rhs) const {
    const double orientation = CalculateOrientation(*origin_ptr_, lhs, rhs);
    if (std::abs(orientation) > kTolerance) {
      return orientation > 0;
    }
    return CalculateSquaredDistance(*origin_ptr_, lhs) < CalculateSquaredDistance(*origin_ptr_, rhs);
  }

 private:
  const Point2D *origin_ptr_;
};

// Структура для потокобезопасной передачи данных
struct ThreadData {
  std::size_t local_lowest;
};

bool FindLowestInSlice(const PointSet &points, int start, int end, std::size_t &local_lowest) {
  local_lowest = static_cast<std::size_t>(start);

  for (int i = start + 1; i <= end; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    if (points[idx].coord_y < points[local_lowest].coord_y ||
        (std::abs(points[idx].coord_y - points[local_lowest].coord_y) < kTolerance &&
         points[idx].coord_x < points[local_lowest].coord_x)) {
      local_lowest = idx;
    }
  }
  return true;
}

bool CheckPointsIdenticalSlice(const PointSet &points, const Point2D &reference, int start, int end,
                               bool &local_identical) {
  for (int i = start; i <= end; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    if (std::abs(points[idx].coord_x - reference.coord_x) > kTolerance ||
        std::abs(points[idx].coord_y - reference.coord_y) > kTolerance) {
      local_identical = false;
      return false;  // Можно прервать, если нашли отличие
    }
  }
  return true;
}

}  // namespace

PetersonGrahamScannerOMP::PetersonGrahamScannerOMP(const InputValue &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

void PetersonGrahamScannerOMP::LoadPoints(const PointSet &points) {
  input_points_ = points;
  external_data_provided_ = true;
}

PointSet PetersonGrahamScannerOMP::GetConvexHull() const {
  return hull_points_;
}

bool PetersonGrahamScannerOMP::ValidationImpl() {
  return GetInput() >= 0;
}

bool PetersonGrahamScannerOMP::PreProcessingImpl() {
  hull_points_.clear();

  if (!external_data_provided_) {
    input_points_.clear();
    const int count = GetInput();
    if (count <= 0) {
      return true;
    }

    input_points_.resize(count);
    const double angle_step = 2.0 * std::numbers::pi / count;

// Параллельное создание точек с default(none)
#pragma omp parallel for default(none) shared(input_points_, count, angle_step)
    for (int i = 0; i < count; ++i) {
      const double angle = angle_step * i;
      input_points_[i] = Point2D(std::cos(angle), std::sin(angle));
    }
  }

  return true;
}

bool PetersonGrahamScannerOMP::AreAllPointsIdentical(const PointSet &points) {
  if (points.empty()) {
    return true;
  }

  const Point2D &reference = points[0];
  const int num_points = static_cast<int>(points.size());

  bool all_identical = true;
  bool error_flag = false;

// Параллельная проверка с default(none)
#pragma omp parallel for default(none) shared(points, reference, num_points, kTolerance, all_identical, error_flag)
  for (int i = 1; i < num_points; ++i) {
    if (error_flag) {
      continue;
    }

    if (std::abs(points[i].coord_x - reference.coord_x) > kTolerance ||
        std::abs(points[i].coord_y - reference.coord_y) > kTolerance) {
#pragma omp atomic write
      all_identical = false;
#pragma omp atomic write
      error_flag = true;
    }
  }

  return all_identical;
}

bool PetersonGrahamScannerOMP::RunImpl() {
  hull_points_.clear();
  const int total_points = static_cast<int>(input_points_.size());

  if (total_points == 0) {
    return true;
  }

  if (AreAllPointsIdentical(input_points_)) {
    hull_points_.push_back(input_points_.front());
    return true;
  }

  if (total_points < 3) {
    hull_points_ = input_points_;
    return true;
  }

  // Параллельный поиск самой нижней точки
  const int num_threads = omp_get_max_threads();
  const int chunk_size = std::max(1, total_points / num_threads);

  std::vector<std::size_t> thread_results(num_threads, 0);
  bool error_flag = false;

#pragma omp parallel default(none) shared(input_points_, total_points, chunk_size, thread_results, error_flag)
  {
    const int tid = omp_get_thread_num();
    const int start = tid * chunk_size;
    const int end = (tid == num_threads - 1) ? (total_points - 1) : (start + chunk_size - 1);

    if (start <= end && start < total_points) {
      std::size_t local_lowest = static_cast<std::size_t>(start);

      for (int i = start + 1; i <= end; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        if (input_points_[idx].coord_y < input_points_[local_lowest].coord_y ||
            (std::abs(input_points_[idx].coord_y - input_points_[local_lowest].coord_y) < kTolerance &&
             input_points_[idx].coord_x < input_points_[local_lowest].coord_x)) {
          local_lowest = idx;
        }
      }

      thread_results[tid] = local_lowest;
    }
  }

  // Находим минимальный среди результатов потоков
  std::size_t lowest_idx = thread_results[0];
  for (int t = 1; t < num_threads; ++t) {
    if (thread_results[t] < static_cast<std::size_t>(total_points)) {
      if (input_points_[thread_results[t]].coord_y < input_points_[lowest_idx].coord_y ||
          (std::abs(input_points_[thread_results[t]].coord_y - input_points_[lowest_idx].coord_y) < kTolerance &&
           input_points_[thread_results[t]].coord_x < input_points_[lowest_idx].coord_x)) {
        lowest_idx = thread_results[t];
      }
    }
  }

  std::swap(input_points_[0], input_points_[lowest_idx]);

  // Сортировка по полярному углу (последовательно для простоты)
  const Point2D origin = input_points_[0];
  PointComparator comp(&origin);
  std::sort(input_points_.begin() + 1, input_points_.end(), comp);

  // Последовательный алгоритм Грэхема
  std::vector<Point2D> stack;
  stack.reserve(total_points);
  stack.push_back(input_points_[0]);
  stack.push_back(input_points_[1]);

  for (int i = 2; i < total_points; ++i) {
    while (static_cast<int>(stack.size()) >= 2) {
      const Point2D &second_last = stack[stack.size() - 2];
      const Point2D &last = stack.back();

      if (CalculateOrientation(second_last, last, input_points_[i]) <= kTolerance) {
        stack.pop_back();
      } else {
        break;
      }
    }
    stack.push_back(input_points_[i]);
  }

  hull_points_ = std::move(stack);
  return true;
}

bool PetersonGrahamScannerOMP::PostProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

double PetersonGrahamScannerOMP::ComputeOrientation(const Point2D &origin, const Point2D &a, const Point2D &b) {
  return CalculateOrientation(origin, a, b);
}

double PetersonGrahamScannerOMP::ComputeDistanceSq(const Point2D &p1, const Point2D &p2) {
  return CalculateSquaredDistance(p1, p2);
}

}  // namespace peterson_r_graham_scan_omp
