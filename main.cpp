#include <torch/torch.h>
#include "inference.h"

int main() {
    //auto test_loader = CSVParser("../results/test_mod_v3.csv");
    //auto graph = test_loader.GetGraph();

    inference::InferenceKNN("../results/als_cpp.txt");
}
