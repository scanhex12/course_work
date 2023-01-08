#include <torch/torch.h>
#include "inference.h"
#pragma comment(linker, "/stack:200000000000")

int main() {
    inference::InferenceBFS("../results/ngcf_implicit_als.txt");
    inference::InferenceKNN("../results/ngcf_implicit_als.txt");
}
