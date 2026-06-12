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

        item_sum.assign(items, 0.0f);
        item_count.assign(items, 0);
        item_score.assign(items, global_mean);
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty() || items <= 0) {
            return;
        }

        for (const Rating& r : incremental_batch) {
            if (static_cast<unsigned>(r.user) >= static_cast<unsigned>(users) ||
                static_cast<unsigned>(r.item) >= static_cast<unsigned>(items)) {
                continue;
            }
            item_sum[r.item] += r.rating - global_mean;
            ++item_count[r.item];
        }

        for (int item = 0; item < items; ++item) {
            if (item_count[item] == 0) {
                item_score[item] = global_mean;
                continue;
            }
            const float score = global_mean +
                                item_sum[item] /
                                    (static_cast<float>(item_count[item]) + item_shrink);
            item_score[item] = std::min(5.0f, std::max(0.5f, score));
        }
    }

    float predict(int user_id, int item_id) {
        if (static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
            static_cast<unsigned>(item_id) >= static_cast<unsigned>(items)) {
            return global_mean;
        }
        return item_score[item_id];
    }

private:
    static constexpr float item_shrink = 5.0f;

    int users = 0;
    int items = 0;
    float global_mean = 0.0f;

    std::vector<float> item_sum;
    std::vector<float> item_score;
    std::vector<int> item_count;
};
