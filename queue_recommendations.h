#include <vector>
#include <queue>
#include <algorithm>


class BFSRecommendations {
public:
    BFSRecommendations(const std::vector<std::vector<size_t>> &g,
                       const std::vector<std::vector<double>> &embeddings) :
            g_(g), embeddings_(embeddings) {
        used_.resize(embeddings_.size());
    }

    std::vector<size_t> GetNeighbours(size_t vertex, size_t radius, size_t limit_answer) {
        cnt_called_++;
        std::queue<std::pair<size_t, size_t>> q;
        q.push({vertex, 0});
        used_[vertex] = cnt_called_;
        std::vector<size_t> answer;
        while (!q.empty() && answer.size() < limit_answer) {
            size_t cur_v = q.front().first;
            size_t cur_d = q.front().second;
            q.pop();
            if (cur_d > radius) {
                continue;
            }
            if (cur_d > 1) {
                answer.push_back(cur_v);
            }
            for (auto u: g_[cur_v]) {
                if (u < used_.size() && used_[u] != cnt_called_) {
                    used_[u] = cnt_called_;
                    q.push({u, cur_d + 1});
                }
            }
        }
        return answer;
    }

    double GetDistance(size_t i, size_t j) {
        double sum = 0;
        for (size_t d = 0; d < embeddings_[0].size(); ++d) {
            sum += (embeddings_[i][d] - embeddings_[j][d]) * (embeddings_[i][d] - embeddings_[j][d]);
        }
        return sum;
    }

    std::vector<size_t> GetNearest(size_t v0, size_t k, size_t r, size_t limit_answer) {
        auto neighbours = GetNeighbours(v0, r, limit_answer);
        std::vector<size_t> answer;
        std::vector<std::pair<double, size_t>> ranked_answer;
        for (auto v: neighbours) {
            ranked_answer.push_back({GetDistance(v0, v), v});
        }
        std::sort(ranked_answer.begin(), ranked_answer.end());
        for (size_t i = 0; i < std::min(k, ranked_answer.size()); ++i) {
            answer.push_back(ranked_answer[i].second);
        }
        return answer;
    }

private:
    size_t cnt_called_ = 0;
    std::vector<size_t> used_;
    const std::vector<std::vector<size_t>>& g_;
    const std::vector<std::vector<double>>& embeddings_;
};