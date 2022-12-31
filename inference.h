#include "fastknn.h"
#include "embedding_parser.h"
#include "logger.h"
#include "csv_parser.h"
#include <iostream>


namespace inference {
    void InferenceKNN(const std::string &path_embeddings, size_t num_projections = 10) {
        auto embedding_paser = EmbeddingParser(path_embeddings, "");
        auto test_loader = CSVParser("../results/test_mod_v3.csv");
        std::cout << "knn start...\n";
        auto knn = FastKNN(embedding_paser.GetVector(), num_projections);
        std::cout << "knn builded\n";
        auto graph = test_loader.GetGraph();
        auto bar = ProgressBar(graph.size(), graph.size() / 100);
        auto logger = Logger({1, 3, 5, 7, 10, 20, 30, 40});
        for (size_t i = 0; i < graph.size(); ++i) {
            if (graph[i].empty()) {
                continue;
            }
            auto predicted = knn.FindNeighbours(i, graph[i].size(), {i});
            logger.UpdateMetrics({}, graph[i]);
            bar.UpgradeProgress(i);
        }
        logger.PrintResult();
    }
}
