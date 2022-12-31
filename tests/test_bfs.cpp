#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../queue_recommendations.h"
#include <vector>

TEST_CASE("bfs 1") {
    std::vector<std::vector<size_t>> g = {{1,2,3},
                                          {4},
                                          {5},
                                          {6},
                                          {},
                                          {},
                                          {}
                                          };
    for (size_t i = 0; i < g.size(); ++i) {
        for (auto u : g[i]) {
            g[u].push_back(i);
        }
    }
    std::vector<std::vector<double>> embeddings = {{0, 0},
                                                   {0, 1},
                                                   {1, 0},
                                                   {1, 1},
                                                   {2, 0},
                                                   {2, 1},
                                                   {3, 0}};

    auto recommender = BFSRecommendations(g, embeddings);
    auto recommendations = recommender.GetNearest(0, 1, 2);
    REQUIRE(recommendations.size() == 1);
    REQUIRE(recommendations[0] == 4);
}
