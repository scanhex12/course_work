#include <iostream>

class ProgressBar {
public:
    ProgressBar(size_t total, size_t ping): ping_(ping), total_(total) {
    }

    void UpgradeProgress(size_t cur_value) {
        if (cur_value % ping_ != 0) {
            return;
        }
        float progress = (cur_value + 0.0) / (total_ + 0.0);
        {
            int barWidth = 70;

            std::cout << "[";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            std::cout.flush();
        }
        std::cout << std::endl;

    }
private:
    size_t total_;
    size_t ping_;
};