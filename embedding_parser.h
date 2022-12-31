#include <map>
#include <torch/script.h>
#include <iostream>
#include <string>
#include "progress_bar.h"

class EmbeddingParser {
public:
    EmbeddingParser(const std::string& path_name, const std::string& attr_name): path_(path_name), attr_(attr_name) {
    }

    std::vector<std::vector<double>> GetPytorchVector() {
        torch::jit::script::Module container = torch::jit::load(path_);
        torch::Tensor weights = container.attr(attr_).toTensor();
        std::vector<std::vector<double>> answer(weights.size(0), std::vector<double>(weights.size(1)));
        for (size_t i = 0; i < answer.size(); ++i) {
            for (size_t j = 0; answer[i].size(); ++j) {
                answer[i][j] = weights[i][j].item<double>();
            }
        }
        return answer;
    }

    std::vector<std::vector<double>> GetVector() {
        std::ifstream myfile;
        myfile.open(path_);
        size_t n, m;
        std::string line;
        std::getline(myfile, line);
        auto nm = SeparateNumbers(line);
        n = nm.first, m = nm.second;
        ProgressBar bar(n, n / 20);
        std::vector<std::vector<double>> ans(n, std::vector<double>(m));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                getline(myfile, line);
                ans[i][j] = std::stoi(line);
            }
            bar.UpgradeProgress(i);
        }
        return ans;
    }
private:
    std::string path_;
    std::string attr_;

    std::pair<size_t, size_t> SeparateNumbers(std::string s) {
        std::string fst, scd;
        bool is_fst = true;
        for (auto elem: s) {
            if (elem == ' ') {
                is_fst = false;
                continue;
            }
            if (is_fst) {
                fst.push_back(elem);
            } else {
                scd.push_back(elem);
            }
        }
        return {std::stoi(fst), std::stoi(scd)};
    }
};
