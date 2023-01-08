#include "fastknn.h"
#include "embedding_parser.h"
#include "logger.h"
#include "csv_parser.h"
#include <iostream>
#include "queue_recommendations.h"
#include <thread>
#include <mutex>
#include <atomic>


namespace inference {
    void InferenceKNN(const std::string &path_embeddings, size_t num_projections = 10) {
        auto embedding_paser = EmbeddingParser(path_embeddings, "");
        auto test_loader = CSVParser("../results/test_mod_v3.csv");
        std::cout << "knn start...\n";
        auto embeddings = embedding_paser.GetVector();
        auto knn = FastKNN(embeddings, num_projections);
        std::cout << "knn builded\n";
        auto graph = test_loader.GetGraph();
        auto bar = ProgressBar(graph.size(), graph.size() / 100);
        std::vector<size_t> k_metrics;
        for (size_t i = 1; i <= 20; ++i) {
            k_metrics.push_back(i);
        }
        auto logger = Logger(k_metrics);
        for (size_t i = 0; i < embeddings.size(); ++i) {
            if (graph[i].empty()) {
                continue;
            }

            auto predicted = knn.FindNeighbours(i, 10, {i});
            logger.UpdateMetrics(predicted, graph[i]);
            bar.UpgradeProgress(i);

        }
        logger.PrintResult();
    }

    void InferenceBFS(const std::string &path_embeddings) {
        auto embedding_paser = EmbeddingParser(path_embeddings, "");
        auto test_loader = CSVParser("../results/test_mod_v3.csv");
        auto train_loader = CSVParser("../results/train_mod_v3.csv");
        auto train_graph = train_loader.GetGraph();
        auto graph = test_loader.GetGraph();
        while (train_graph.size() < graph.size()) {
            train_graph.push_back({});
        }

        auto embeddings = embedding_paser.GetVector();
        auto recommender = BFSRecommendations(train_graph, embeddings);
        auto bar = ProgressBar(graph.size(), graph.size() / 100);
        std::vector<size_t> k_metrics;
        for (size_t i = 1; i <= 20; ++i) {
            k_metrics.push_back(i);
        }
        auto logger = Logger(k_metrics);
        size_t n_threads = std::thread::hardware_concurrency() - 2;
        std::mutex mtx;
        std::vector<std::thread> threads_processor(n_threads);
        int atom_len = (embeddings.size() + n_threads - 1) / n_threads;
        for (int k = 0; k < embeddings.size(); k += atom_len) {
            threads_processor[k / atom_len] = std::thread(
                    [atom_len, &mtx, &recommender, &logger, &embeddings, &graph](size_t k) {
                        for (size_t i = k; i < std::min(k + atom_len, embeddings.size()); ++i) {
                            if (graph[i].empty()) {
                                continue;
                            }
                            auto predicted = recommender.GetNeighbours(i, 3, 100);
                            {
                                std::lock_guard<std::mutex> lock(mtx);
                                logger.UpdateMetrics(predicted, graph[i]);
                            }
                        }
                    }, k);
        }
        for (size_t i = 0; i < n_threads; ++i) {
            threads_processor[i].join();
        }

        logger.PrintResult();
    }
}
