#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../embedding_parser.h"


TEST_CASE("weights") {
    auto parser = EmbeddingParser("../results/container_simple.pt", "als");
    auto vec = parser.GetVector();
    REQUIRE(vec.size() > 1000);
    REQUIRE(vec[0].size() == 64);
}