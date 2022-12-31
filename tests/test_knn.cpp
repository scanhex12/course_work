#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../fastknn.h"
#include <unordered_set>


TEST_CASE("1-d knn") {
    std::vector<std::vector<double>> embedings = {{1}, {2}, {5}};
    auto knn = FastKNN(embedings, 1);
    std::unordered_set<size_t> q;
    auto answer1 = knn.FindNeighbours(1, 1, q);
    REQUIRE(answer1.size() == 1);
    REQUIRE(answer1[0] == 0);
}


TEST_CASE("1-d knn smarter") {
    std::vector<std::vector<double>> embedings = {{1}, {8}, {2}, {10}, {5}, {6}};
    auto knn = FastKNN(embedings, 1);
    std::unordered_set<size_t> q;
    auto answer1 = knn.FindNeighbours(1, 3, q); //3, 5, 4
    REQUIRE(answer1.size() == 3);
    std::sort(answer1.begin(), answer1.end());
    REQUIRE(answer1[0] == 3);
    REQUIRE(answer1[1] == 4);
    REQUIRE(answer1[2] == 5);
}

TEST_CASE("2-d knn") {
    std::vector<std::vector<double>> embedings = {{0, 0},
                                                  {0, 1},
                                                  {1, 0},
                                                  {1, 1},
                                                  {2, 1}};
    auto knn = FastKNN(embedings, 10);
    std::unordered_set<size_t> q;
    auto answer1 = knn.FindNeighbours(3, 3, q);
    REQUIRE(answer1.size() == 3);
    std::sort(answer1.begin(), answer1.end());
    REQUIRE(answer1[0] == 1);
    REQUIRE(answer1[1] == 2);
    REQUIRE(answer1[2] == 4);
}