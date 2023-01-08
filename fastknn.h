#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <iostream>

class FastKNN {
public:
    FastKNN(const std::vector<std::vector<double>> &embedings, size_t number_projections) :
            embeddings_(embedings) {
        size_t num_elements = embeddings_.size();
        size_t dimension = embeddings_[0].size();

        random_projections_ = std::vector<std::vector<double>>(number_projections, std::vector<double>(dimension));

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> distribution{0, 1};


        for (size_t i = 0; i < number_projections; ++i) {
            for (size_t j = 0; j < dimension; ++j) {
                random_projections_[i][j] = distribution(gen);
            }
        }

        sorted_projections_.resize(number_projections);
        for (size_t ind_projection = 0; ind_projection < number_projections; ++ind_projection) {
            std::vector<double> values_projections(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                for (size_t j = 0; j < dimension; ++j) {
                    values_projections[i] += random_projections_[ind_projection][j] * embedings[i][j];
                }
                sorted_projections_[ind_projection].push_back({values_projections[i], i});
            }
            std::sort(sorted_projections_[ind_projection].begin(), sorted_projections_[ind_projection].end());
            sorted_values_projections_.push_back(values_projections);
            unsorted_values_projections_.push_back(values_projections);
            std::sort(sorted_values_projections_.back().begin(), sorted_values_projections_.back().end());
        }
    }

    std::vector<size_t>
    FindNeighbours(size_t request_ind, size_t k, const std::unordered_set<size_t> &blocked_candidates) {
        std::unordered_set<size_t> candidates;
        for (size_t ind_projection = 0; ind_projection < sorted_projections_.size(); ++ind_projection) {
            size_t mid_position = std::lower_bound(sorted_values_projections_[ind_projection].begin(),
                                                   sorted_values_projections_[ind_projection].end(),
                                                   unsorted_values_projections_[ind_projection][request_ind]) -
                                  sorted_values_projections_[ind_projection].begin();

            size_t left_position = std::max(0, static_cast<int>(mid_position) - (static_cast<int>(k) / 2));
            size_t right_position = std::min(unsorted_values_projections_[ind_projection].size(),
                                             left_position + k + 1);
            for (size_t pos = left_position; pos < right_position; ++pos) {
                if (!blocked_candidates.count(sorted_projections_[ind_projection][pos].second) &&
                    sorted_projections_[ind_projection][pos].second != request_ind) {
                    candidates.insert(sorted_projections_[ind_projection][pos].second);
                }
            }
        }
        std::vector<std::pair<double, size_t>> ranked_candidates;
        for (auto candidate: candidates) {
            double rank = 0;
            for (size_t j = 0; j < embeddings_[0].size(); ++j) {
                rank += (embeddings_[candidate][j] - embeddings_[request_ind][j]) *
                        (embeddings_[candidate][j] - embeddings_[request_ind][j]);
            }
            ranked_candidates.push_back({rank, candidate});
        }
        std::sort(ranked_candidates.begin(), ranked_candidates.end());
        std::vector<size_t> answer;
        for (size_t i = 0; i < std::min(k, ranked_candidates.size()); ++i) {
            answer.push_back(ranked_candidates[i].second);
        }
        return answer;

    }

private:
    std::vector<std::vector<double>> embeddings_;
    std::vector<std::vector<double>> random_projections_;
    std::vector<std::vector<std::pair<double, size_t>>> sorted_projections_;
    std::vector<std::vector<double>> sorted_values_projections_;
    std::vector<std::vector<double>> unsorted_values_projections_;
};