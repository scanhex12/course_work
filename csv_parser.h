#include <string>
#include <vector>
#include <algorithm>


class CSVParser {
public:
    CSVParser(const std::string &path) :
            path_(path) {
    }

    std::vector<std::pair<size_t, size_t>> GetEdgeList() {
        std::ifstream myfile;
        myfile.open(path_);
        std::string line;
        std::getline(myfile, line);
        std::vector<std::pair<size_t, size_t>> answer;
        while (std::getline(myfile, line)) {
            std::vector<std::string> parsed_line = Split(line);
            size_t u = std::stoi(parsed_line[0]);
            size_t v = std::stoi(parsed_line[1]);
            answer.push_back({u, v});
        }
        return answer;
    }

    std::vector<std::vector<size_t>> GetGraph() {
        std::ifstream myfile;
        myfile.open(path_);
        std::string line;
        std::getline(myfile, line);
        size_t index_v = GetIndex(line);
        std::vector<std::vector<size_t>> graph;
        ProgressBar bar = ProgressBar(3349019, 3349019 / 10);
        size_t i = 0;
        while (std::getline(myfile, line)) {
            std::vector<std::string> parsed_line = Split(line, ",");
            size_t u = std::stoi(parsed_line[index_v]);
            size_t v = std::stoi(parsed_line[index_v + 1]);
            while (graph.size() < std::max(u, v) + 1) {
                graph.push_back({});
            }
            graph[u].push_back(v);
            graph[v].push_back(u);
            i += 1;
            bar.UpgradeProgress(i);
        }
        return graph;
    }

private:
    std::string path_;

    std::vector<std::string> Split(const std::string &strin, const std::string &delimiter = " ") {
        std::vector<std::string> answer;
        std::string back;
        for (size_t i = 0; i < strin.size();) {
            if (i < strin.size() - delimiter.size() + 1 &&
                strin.substr(i, delimiter.size()) == delimiter) {
                answer.push_back(back);
                back = "";
                i += delimiter.size();
            } else {
                back.push_back(strin[i]);
                i++;
            }
        }
        answer.push_back(back);
        return answer;
    }

    size_t GetIndex(const std::string &s) {
        size_t answer = 0;
        for (auto elem:s) {
            if (elem == ',') {
                answer++;
            }
            if (elem == 'u') {
                return answer;
            }
        }
        return 0;
    }
};