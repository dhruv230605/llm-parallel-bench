#include <algorithm>
#include <future>
#include <vector>

namespace {

void quicksort_impl(std::vector<int> &values, int left, int right, int depth) {
    if (left >= right) {
        return;
    }

    int i = left;
    int j = right;
    const int pivot = values[left + (right - left) / 2];
    while (i <= j) {
        while (values[i] < pivot) {
            ++i;
        }
        while (values[j] > pivot) {
            --j;
        }
        if (i <= j) {
            std::swap(values[i], values[j]);
            ++i;
            --j;
        }
    }

    const bool spawn = depth < 3 && (right - left) > 2048;
    std::future<void> left_future;
    if (left < j) {
        if (spawn) {
            left_future = std::async(std::launch::async, [&values, left, j, depth]() {
                quicksort_impl(values, left, j, depth + 1);
            });
        } else {
            quicksort_impl(values, left, j, depth + 1);
        }
    }
    if (i < right) {
        quicksort_impl(values, i, right, depth + 1);
    }
    if (left_future.valid()) {
        left_future.get();
    }
}

}  // namespace

void parallel_quicksort(std::vector<int> &values) {
    if (values.empty()) {
        return;
    }
    quicksort_impl(values, 0, static_cast<int>(values.size()) - 1, 0);
}
