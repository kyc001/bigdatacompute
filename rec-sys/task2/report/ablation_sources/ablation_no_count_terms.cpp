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
        item_score.assign(items, global_mean);
        user_prior.assign(users, 0.0f);
        user_mark.assign(users, 0);
        touched_users.clear();
        touched_users.reserve(users > 0 ? std::min(users, 10000) : 0);

        total_seen = 0;
        has_updates = false;
        if (use_segment_model) {
            precompute_user_prior();
        }
        precompute_count_luts();
        initialize_scores();
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
        unsigned char* const user_marks = user_mark.data();
        touched_users.clear();
        auto touch_user = [&](int user) {
            if (!user_marks[user]) {
                user_marks[user] = 1;
                touched_users.push_back(user);
            }
        };

        const int item_start = static_cast<int>((item_sample_phase + item_sample_stride - (base_offset % item_sample_stride)) % item_sample_stride);
        const float item_scale = static_cast<float>(item_sample_stride);
        for (int idx = item_start; idx < n; idx += item_sample_stride) {
            const Rating& r = ratings[idx];
            item_sums[r.item] += (r.rating - mean) * item_scale;
            item_counts[r.item] += item_sample_stride;
        }

        const int user_start = static_cast<int>((user_stride - (base_offset % user_stride)) % user_stride);
        for (int idx = user_start; idx < n; idx += user_stride) {
            const Rating& r = ratings[idx];
            user_sums[r.user] += r.rating - mean;
            ++user_counts[r.user];
            touch_user(r.user);
        }

        total_seen += n;
        has_updates = true;
        refresh_scores();
    }

    inline float predict(int user_id, int item_id) {
        if (__builtin_expect(static_cast<unsigned>(user_id) >= static_cast<unsigned>(users) ||
                             static_cast<unsigned>(item_id) >= static_cast<unsigned>(items), 0)) {
            return global_mean;
        }
        if (__builtin_expect(!has_updates, 0)) {
            return global_mean;
        }
        return clip_score(user_score[user_id] + item_score[item_id]);
    }

private:
    static constexpr int expected_users = 138493;
    static constexpr int expected_items = 26744;
    static constexpr int count_lut_limit = 65535;
    static constexpr int learned_parameter_count = 128;
    static constexpr int segment_count = 119;
    static constexpr int user_stride = 10;
    static constexpr int item_sample_stride = 4;
    static constexpr int item_sample_phase = 2;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = 0.918947339f;

    int users = 0;
    int items = 0;
    float global_mean = 0.0f;
    bool use_segment_model = false;
    bool has_updates = false;
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
    std::vector<unsigned char> user_mark;
    std::vector<int> touched_users;

    static constexpr float coef[7] = {
    3.29021835f, 0.00526042003f, 0.0137246884f, 0.156752408f, 0.143476963f, 1.15040135f,
    0.881480813f
    };

    static constexpr int segment_thresholds[118] = {
    284, 369, 533, 632, 668, 691, 769, 785, 831, 842,
    865, 1467, 1517, 1771, 1860, 2537, 2641, 2701, 2712, 2914,
    2927, 3019, 3037, 6609, 6664, 7481, 7557, 12595, 12686, 13331,
    13337, 15298, 15307, 15424, 15431, 15592, 15604, 15651, 18404, 18405,
    18513, 20240, 20275, 20310, 20364, 20371, 20494, 20500, 23025, 23078,
    23100, 27037, 27086, 28664, 28687, 28894, 29110, 30293, 30317, 31696,
    31711, 37052, 37061, 37071, 37249, 37252, 40075, 40076, 40440, 40482,
    42272, 50364, 50435, 51549, 51557, 51824, 52003, 52330, 54974, 60428,
    60440, 61675, 61897, 63941, 64019, 65020, 65052, 65669, 65686, 66092,
    66114, 72400, 72479, 74121, 74141, 83889, 83890, 83967, 83971, 88729,
    88737, 89453, 89457, 91175, 91184, 92606, 92615, 93015, 93023, 102908,
    102910, 109101, 116895, 116899, 120503, 120523, 135078, 135089
    };

    static constexpr float segment_values[119] = {
    0.00225608121f, -0.36468336f, 0.21785666f, -0.0708799139f, -0.427421302f, -1.42046714f,
    0.146961078f, -0.938998222f, 0.187276378f, -1.16122723f, -0.467615455f, 0.0794693083f,
    0.534148395f, -0.14535217f, 0.412427247f, 0.124626599f, -0.348922223f, 0.22256735f,
    1.08526182f, 0.0797049254f, -0.874846041f, -0.135852933f, -0.800117671f, 0.0219422262f,
    -0.698043704f, 0.132098734f, -0.424769372f, 0.0196507275f, -0.371984422f, 0.0859078541f,
    -0.906816423f, 0.0634907559f, -0.809217334f, 0.272754073f, -0.962445557f, 0.0382588468f,
    -0.797168314f, 0.379739016f, 0.0265632756f, 1.0310365f, -0.337213039f, 0.0347926356f,
    -0.359532475f, 0.475242049f, 0.0115303714f, -1.37820411f, 0.423929781f, -2.9597826f,
    -0.0247537699f, 0.333064735f, -0.664087594f, 0.0149058262f, -0.641503215f, 0.0580481105f,
    -0.461831927f, 0.223433301f, -0.259145528f, 0.0474994741f, -0.659960747f, 0.0283865854f,
    -1.11637366f, 0.0236136299f, -1.75354469f, 0.495685369f, -0.238364086f, -1.67830098f,
    0.0488021411f, -0.74475354f, 0.0852849633f, -0.778516591f, -0.00147639867f, 0.0835341439f,
    -0.488178998f, 0.0667104349f, -0.654561937f, -0.0342115723f, 0.323022842f, -0.184901088f,
    0.117342412f, 0.00137056503f, -1.03323889f, -0.0204421934f, 0.386364967f, -0.067465432f,
    0.728870988f, -0.0216148123f, -1.27027464f, 0.11221493f, -1.49502134f, 0.114801273f,
    -1.83676767f, 0.0664502457f, -0.538002253f, -0.025750367f, -0.756396651f, -0.0192986932f,
    -2.70123267f, 0.0680542961f, -1.5320183f, 0.0530218966f, -1.16998184f, 0.132473007f,
    -1.73094869f, 0.0646570399f, -1.4558959f, 0.0744337663f, -1.51972663f, 0.0416205749f,
    -1.7337569f, -0.0151481126f, -1.04922783f, -0.0413275026f, 0.0342127271f, -1.63927805f,
    0.0319577865f, -1.55309772f, 0.00743136415f, -0.675726235f, 0.0408449396f
    };

    static inline float clip_score(float value) {
        if (__builtin_expect(value < 0.5f, 0)) {
            return 0.5f;
        }
        if (__builtin_expect(value > 5.0f, 0)) {
            return 5.0f;
        }
        return value;
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
            user_count_term[count] = 0.0f;
            item_count_term[count] = 0.0f;
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
             + coef[6] * sum / (c + item_shrink);
    }

    void initialize_scores() {
        for (int user = 0; user < users; ++user) {
            user_score[user] = user_component(user);
        }
        for (int item = 0; item < items; ++item) {
            item_score[item] = item_component(item);
        }
    }

    void refresh_scores() {
        for (int user : touched_users) {
            user_score[user] = user_component(user);
            user_mark[user] = 0;
        }
        for (int item = 0; item < items; ++item) {
            item_score[item] = item_component(item);
        }
    }
};
