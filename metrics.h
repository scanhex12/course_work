#include <vector>
#include <unordered_set>
#include <cmath>

namespace metrics {
    double APK(const std::vector <size_t> &predictions, const std::vector <size_t> &target, size_t k) {
        std::unordered_set <size_t> used;
        std::unordered_set <size_t> target_set;
        for (auto elem: target) {
            target_set.insert(elem);
        }
        double hints = 0, total = 0, score = 0;
        for (size_t i = 0; i < std::min(predictions.size(), k); ++i) {
            if (target_set.count(predictions[i]) && !used.count(predictions[i])) {
                hints += 1;
                score += hints / (i + 1.0);
            }
        }
        return score / (0.0 + std::min(k, target.size()));
    }

    double MAPK(const std::vector <std::vector<size_t>> &predictions,
                const std::vector <std::vector<size_t>> &target,
                size_t K) {
        double sum_apk = 0;
        double cnt_all = 0;
        for (size_t i = 0; i < target.size(); ++i) {
            if (target[i].empty()) {
                continue;
            }
            cnt_all += 1.0;
            sum_apk += APK(predictions[i], target[i], K);
        }
        return sum_apk / cnt_all;
    }

    double MR(const std::vector <size_t> &predictions, const std::vector <size_t> &target) {
        std::unordered_set <size_t> used;
        std::unordered_set <size_t> target_set;
        for (auto elem: target) {
            target_set.insert(elem);
        }
        double total = 0, score = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (target_set.count(predictions[i]) && !used.count(predictions[i])) {
                score += 1.0 / (i + 1.0);
            }
        }
        return score;
    }

    double MRR(const std::vector <std::vector<size_t>> &predictions,
               const std::vector <std::vector<size_t>> &target) {
        double sum_mrr = 0;
        double cnt_all = 0;
        for (size_t i = 0; i < target.size(); ++i) {
            if (target[i].empty()) {
                continue;
            }
            cnt_all += 1.0;
            sum_mrr += MR(predictions[i], target[i]);
        }
        return sum_mrr / cnt_all;
    }

    double NDCGUnique(const std::vector <size_t> &predictions,
                      const std::vector <size_t> &target) {
        std::unordered_set <size_t> used;
        std::unordered_set <size_t> target_set;
        for (auto elem: target) {
            target_set.insert(elem);
        }
        double total = 0, score = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (target_set.count(predictions[i]) && !used.count(predictions[i])) {
                score += 1.0 / log2(i + 2.0);
            }
        }
        return score;
    }

    double NDCG(const std::vector <std::vector<size_t>> &predictions,
                const std::vector <std::vector<size_t>> &target) {
        double sum_mrr = 0;
        double cnt_all = 0;
        for (size_t i = 0; i < target.size(); ++i) {
            if (target[i].empty()) {
                continue;
            }
            cnt_all += 1.0;
            sum_mrr += NDCGUnique(predictions[i], target[i]);
        }
        return sum_mrr / cnt_all;
    }

    double RecallKUnique(const std::vector <size_t> &predictions, const std::vector <size_t> &target, size_t k) {
        std::unordered_set <size_t> used;
        std::unordered_set <size_t> target_set;
        for (auto elem: target) {
            target_set.insert(elem);
        }
        double total = 0, score = 0;
        for (size_t i = 0; i < std::min(predictions.size(), k); ++i) {
            if (target_set.count(predictions[i]) && !used.count(predictions[i])) {
                score += 1.0 / (target.size() + 0.0);
            }
        }
        return score;
    }

    double PrecisionK(const std::vector <std::vector<size_t>> &predictions,
                const std::vector <std::vector<size_t>> &target,
                size_t K) {
        double sum_apk = 0;
        double cnt_all = 0;
        for (size_t i = 0; i < target.size(); ++i) {
            if (target[i].empty()) {
                continue;
            }
            cnt_all += 1.0;
            sum_apk += RecallKUnique(predictions[i], target[i], K);
        }
        return sum_apk / cnt_all;
    }
}