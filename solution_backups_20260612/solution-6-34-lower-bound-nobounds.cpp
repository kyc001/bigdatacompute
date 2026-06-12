#include <algorithm>
#include <vector>

struct Rating {
    int user;
    int item;
    float rating;
};

class IncrementalSVD {
public:
    void load_base_model(float*, float*, int u_size, int i_size, int, float mean) {
        users = std::max(0, u_size);
        items = std::max(0, i_size);
        global_mean = mean;
    }

    void update(const std::vector<Rating>&) {
    }

    inline float predict(int, int) {
        return global_mean;
    }

private:
    int users = 0;
    int items = 0;
    float global_mean = 0.0f;
};
