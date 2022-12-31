#include <vector>
#include "metrics.h"
#include <iostream>
#include <map>
#include <string>


class Logger {
public:
    Logger(const std::vector<size_t>& k): k_(k){
    }

    void UpdateMetrics(const std::vector<size_t>& predictions, const std::vector<size_t>& target) {
        for (size_t i = 0; i < names_metrics.size(); ++i) {
            if (i == 0 || i == 3) {
                for (auto elem: k_) {
                    std::string cur_name_metric = names_metrics[i] + std::to_string(elem);
                    cnt_metrics_[cur_name_metric] += 1;
                    if (i == 0) {
                        sum_metrics_[cur_name_metric] += metrics::MAPK({predictions}, {target}, elem);
                    } else {
                        sum_metrics_[cur_name_metric] += metrics::PrecisionK({predictions}, {target}, elem);
                    }
                }
            } else {
                std::string cur_name_metric = names_metrics[i];
                cnt_metrics_[cur_name_metric] += 1;
                if (i == 1) {
                    sum_metrics_[cur_name_metric] += metrics::MRR({predictions}, {target});
                } else {
                    sum_metrics_[cur_name_metric] += metrics::NDCG({predictions}, {target});
                }
            }

        }
    }

    void PrintResult() {
        std::cout << "=========RESULTS=========\n";
        for (auto elem: sum_metrics_) {
            std::cout << elem.first << ": " << (elem.second) / cnt_metrics_[elem.first] << '\n';
        }
        std::cout << "=========================\n";
    }

private:
    std::vector<size_t> k_;

    std::map<std::string, double> sum_metrics_;
    std::map<std::string, double> cnt_metrics_;

    const std::vector<std::string> names_metrics = {
            "MAP@",
            "MRR",
            "NDCG",
            "Precision@"
    };
};