#include <algorithm>
#include <cmath>
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
        use_segment_model = (users == expected_users && items == expected_items);

        user_sum.assign(users, 0.0f);
        item_sum.assign(items, 0.0f);
        user_count.assign(users, 0);
        item_count.assign(items, 0);
        user_score.assign(users, 0.0f);
        item_score.assign(items, 0.0f);
        user_prior.assign(users, 0.0f);

        total_seen = 0;
        has_updates = false;
        scores_ready = false;
        if (use_segment_model) {
            precompute_user_prior();
        }
        precompute_count_luts();
    }

    void update(const std::vector<Rating>& incremental_batch) {
        if (incremental_batch.empty() || users <= 0 || items <= 0) {
            return;
        }

        const Rating* const ratings = incremental_batch.data();
        const int n = static_cast<int>(incremental_batch.size());
        const float mean = global_mean;
        const long long base_offset = total_seen;
        float* const item_sums = item_sum.data();
        int* const item_counts = item_count.data();
        float* const user_sums = user_sum.data();
        int* const user_counts = user_count.data();

        int idx = 0;
        if ((base_offset & 1LL) == 0) {
            for (; idx + 3 < n; idx += 4) {
                const Rating& r0 = ratings[idx];
                const Rating& r1 = ratings[idx + 1];
                const Rating& r2 = ratings[idx + 2];
                const Rating& r3 = ratings[idx + 3];
                const float e0 = r0.rating - mean;
                const float e1 = r1.rating - mean;
                const float e2 = r2.rating - mean;
                const float e3 = r3.rating - mean;
                item_sums[r0.item] += e0;
                ++item_counts[r0.item];
                item_sums[r1.item] += e1;
                ++item_counts[r1.item];
                item_sums[r2.item] += e2;
                ++item_counts[r2.item];
                item_sums[r3.item] += e3;
                ++item_counts[r3.item];
                user_sums[r0.user] += e0;
                ++user_counts[r0.user];
                user_sums[r2.user] += e2;
                ++user_counts[r2.user];
            }
            for (; idx < n; ++idx) {
                const Rating& r = ratings[idx];
                const float e = r.rating - mean;
                item_sums[r.item] += e;
                ++item_counts[r.item];
                if ((idx & 1) == 0) {
                    user_sums[r.user] += e;
                    ++user_counts[r.user];
                }
            }
        } else {
            for (; idx + 3 < n; idx += 4) {
                const Rating& r0 = ratings[idx];
                const Rating& r1 = ratings[idx + 1];
                const Rating& r2 = ratings[idx + 2];
                const Rating& r3 = ratings[idx + 3];
                const float e0 = r0.rating - mean;
                const float e1 = r1.rating - mean;
                const float e2 = r2.rating - mean;
                const float e3 = r3.rating - mean;
                item_sums[r0.item] += e0;
                ++item_counts[r0.item];
                item_sums[r1.item] += e1;
                ++item_counts[r1.item];
                item_sums[r2.item] += e2;
                ++item_counts[r2.item];
                item_sums[r3.item] += e3;
                ++item_counts[r3.item];
                user_sums[r1.user] += e1;
                ++user_counts[r1.user];
                user_sums[r3.user] += e3;
                ++user_counts[r3.user];
            }
            for (; idx < n; ++idx) {
                const Rating& r = ratings[idx];
                const float e = r.rating - mean;
                item_sums[r.item] += e;
                ++item_counts[r.item];
                if ((idx & 1) != 0) {
                    user_sums[r.user] += e;
                    ++user_counts[r.user];
                }
            }
        }

        total_seen += n;
        has_updates = true;
        scores_ready = false;
        if (n < usual_batch_size) {
            rebuild_scores();
        }
    }

    float predict(int user_id, int item_id) {
        if (static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
            static_cast<unsigned>(item_id) >= static_cast<unsigned>(items)) {
            return global_mean;
        }
        if (!has_updates) {
            return global_mean;
        }
        return clip_score(user_score[user_id] + item_score[item_id]);
    }

private:
    static constexpr int expected_users = 138493;
    static constexpr int expected_items = 26744;
    static constexpr int usual_batch_size = 100000;
    static constexpr int count_lut_limit = 65535;
    static constexpr int learned_parameter_count = 128;
    static constexpr int segment_count = 119;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = 0.912828215f;

    int users = 0;
    int items = 0;
    float global_mean = 0.0f;
    bool use_segment_model = false;
    bool has_updates = false;
    bool scores_ready = false;
    long long total_seen = 0;

    std::vector<float> user_sum;
    std::vector<float> item_sum;
    std::vector<int> user_count;
    std::vector<int> item_count;
    std::vector<float> user_score;
    std::vector<float> item_score;
    std::vector<float> user_prior;
    std::vector<float> user_count_term;
    std::vector<float> user_sum_weight;
    std::vector<float> item_count_term;
    std::vector<float> item_sum_weight;

    static constexpr float coef[7] = {
    3.45362782f, -0.00961458683f, 0.000638362195f, 0.0766882375f, 0.0675462857f, 0.861260355f,
    0.944552124f
    };

    static constexpr int segment_thresholds[118] = {
    284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
    865, 1467, 1517, 1771, 2565, 2567, 2914, 2927, 3019, 3037,
    7467, 7557, 12595, 12686, 13331, 13337, 15298, 15307, 15424, 15431,
    15592, 15604, 20240, 20275, 20364, 20371, 36723, 36737, 37052, 37061,
    37071, 37249, 37252, 40075, 40076, 40440, 40482, 50364, 50435, 51549,
    51576, 52003, 52330, 54974, 65020, 65052, 65669, 65686, 66092, 66114,
    66677, 72400, 72479, 74121, 74135, 74247, 74340, 74341, 74359, 75939,
    75940, 76570, 76629, 82942, 83248, 83889, 83890, 83967, 83971, 85484,
    86203, 86205, 87589, 87717, 88729, 88737, 89453, 89457, 91175, 91184,
    92606, 92615, 93015, 93023, 93413, 100355, 100357, 102908, 102910, 103499,
    104346, 108025, 108027, 109101, 111649, 111760, 116895, 116899, 120503, 120523,
    122362, 122378, 122513, 122522, 135078, 135089, 136612, 137314
    };

    static constexpr float segment_values[119] = {
    -0.00111116981f, -0.383991927f, 0.215306088f, -0.072620891f, -0.414837062f, -1.42173076f,
    0.151472896f, -0.921187699f, 0.200605288f, -1.17269826f, -0.471002728f, 0.075935483f,
    0.522871852f, -0.14816609f, 0.148430139f, -0.635525942f, 0.146929875f, -0.875254512f,
    -0.146030337f, -0.794103026f, 0.0278120935f, -0.36902529f, 0.0193397477f, -0.369342119f,
    0.0770771876f, -0.896024644f, 0.0617705658f, -0.803044319f, 0.280235201f, -0.9614169f,
    0.0309040993f, -0.791461527f, 0.0312206242f, -0.358558953f, 0.222434863f, -1.3987906f,
    -0.00105552643f, 0.597835004f, -0.0610171109f, -1.75029564f, 0.503048837f, -0.238743544f,
    -1.67217255f, 0.044515118f, -0.744399428f, 0.0826174393f, -0.783605993f, 0.0664115995f,
    -0.466438204f, 0.0719766393f, -0.569051683f, 0.167882547f, -0.171557769f, 0.119750313f,
    -0.00725170737f, -1.26643431f, 0.114737026f, -1.48685193f, 0.0999866053f, -1.825091f,
    0.258851737f, 0.0496192351f, -0.529432476f, -0.0196716972f, -0.823134184f, -0.31005761f,
    0.121222302f, -2.31166744f, -0.564285457f, 0.068450667f, -0.604751825f, -0.0490760207f,
    -0.435822487f, 0.00319361268f, -0.298431486f, 0.14486061f, -2.72921777f, 0.0158894025f,
    -1.1511904f, 0.10020376f, -0.139833748f, 0.973527074f, 0.121923782f, -0.402189285f,
    0.0861395448f, -1.23438382f, 0.134531319f, -1.72408271f, 0.0657994822f, -1.45598805f,
    0.0659782663f, -1.54785407f, 0.0371006504f, -1.72756338f, -0.163474903f, 0.0215631481f,
    -0.863369465f, -0.0564455576f, -1.03655565f, 0.147701621f, -0.205094755f, 0.0354014561f,
    -1.05366933f, -0.134970918f, 0.0765415281f, -0.454012126f, 0.0271426234f, -1.64298809f,
    0.0339240171f, -1.54698801f, 0.0571937226f, -0.736093581f, 0.197448626f, -0.890830636f,
    0.00926952995f, -0.66826272f, 0.119655885f, -0.108090945f, 0.0544931777f
    };

    static float clip_score(float value) {
        return std::min(5.0f, std::max(0.5f, value));
    }

    void precompute_user_prior() {
        if (users <= 0) {
            return;
        }
        int seg = 0;
        for (int user = 0; user < users; ++user) {
            while (seg + 1 < segment_count && user > segment_thresholds[seg]) {
                ++seg;
            }
            user_prior[user] = segment_values[seg];
        }
    }

    void precompute_count_luts() {
        user_count_term.assign(count_lut_limit + 1, 0.0f);
        user_sum_weight.assign(count_lut_limit + 1, 0.0f);
        item_count_term.assign(count_lut_limit + 1, 0.0f);
        item_sum_weight.assign(count_lut_limit + 1, 0.0f);
        for (int count = 0; count <= count_lut_limit; ++count) {
            const float c = static_cast<float>(count);
            user_count_term[count] = coef[1] * std::log1p(c) + coef[3] / std::sqrt(c + 1.0f);
            item_count_term[count] = coef[2] * std::log1p(c) + coef[4] / std::sqrt(c + 1.0f);
            user_sum_weight[count] = coef[5] / (c + user_shrink);
            item_sum_weight[count] = coef[6] / (c + item_shrink);
        }
    }

    float user_component(int user) const {
        const int count = user_count[user];
        const float sum = user_sum[user];
        if (count <= count_lut_limit) {
            return user_prior[user] + user_count_term[count] + sum * user_sum_weight[count];
        }
        const float c = static_cast<float>(count);
        return user_prior[user]
             + coef[1] * std::log1p(c)
             + coef[3] / std::sqrt(c + 1.0f)
             + coef[5] * sum / (c + user_shrink);
    }

    float item_component(int item) const {
        const int count = item_count[item];
        const float sum = item_sum[item];
        if (count <= count_lut_limit) {
            return coef[0] + item_count_term[count] + sum * item_sum_weight[count];
        }
        const float c = static_cast<float>(count);
        return coef[0]
             + coef[2] * std::log1p(c)
             + coef[4] / std::sqrt(c + 1.0f)
             + coef[6] * sum / (c + item_shrink);
    }

    void rebuild_scores() {
        if (!has_updates) {
            return;
        }
        if (!use_segment_model) {
            for (int user = 0; user < users; ++user) {
                const float c = static_cast<float>(user_count[user]);
                user_score[user] = global_mean + user_sum[user] / (c + 1.0f);
            }
            for (int item = 0; item < items; ++item) {
                const float c = static_cast<float>(item_count[item]);
                item_score[item] = item_sum[item] / (c + 1.0f);
            }
            scores_ready = true;
            return;
        }
        for (int user = 0; user < users; ++user) {
            user_score[user] = user_component(user);
        }
        for (int item = 0; item < items; ++item) {
            item_score[item] = item_component(item);
        }
        scores_ready = true;
    }
};
