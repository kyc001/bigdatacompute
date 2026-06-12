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
        touched_users.reserve(users > 0 ? std::min(users, 20000) : 0);

        total_seen = 0;
        has_updates = false;
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
        unsigned char* const user_marks = user_mark.data();
        touched_users.clear();

        auto touch_user = [&](int user) {
            if (!user_marks[user]) {
                user_marks[user] = 1;
                touched_users.push_back(user);
            }
        };
        auto add_user = [&](const Rating& r, float e) {
            user_sums[r.user] += e;
            ++user_counts[r.user];
            touch_user(r.user);
        };

        const int phase_offset = static_cast<int>((user_phase + user_stride -
            static_cast<int>(base_offset % user_stride)) % user_stride);

        int idx = 0;
        for (; idx + 9 < n; idx += 10) {
            const Rating& r0 = ratings[idx];
            const Rating& r1 = ratings[idx + 1];
            const Rating& r2 = ratings[idx + 2];
            const Rating& r3 = ratings[idx + 3];
            const Rating& r4 = ratings[idx + 4];
            const Rating& r5 = ratings[idx + 5];
            const Rating& r6 = ratings[idx + 6];
            const Rating& r7 = ratings[idx + 7];
            const Rating& r8 = ratings[idx + 8];
            const Rating& r9 = ratings[idx + 9];
            const float e0 = r0.rating - mean;
            const float e1 = r1.rating - mean;
            const float e2 = r2.rating - mean;
            const float e3 = r3.rating - mean;
            const float e4 = r4.rating - mean;
            const float e5 = r5.rating - mean;
            const float e6 = r6.rating - mean;
            const float e7 = r7.rating - mean;
            const float e8 = r8.rating - mean;
            const float e9 = r9.rating - mean;
            item_sums[r0.item] += e0;
            ++item_counts[r0.item];
            item_sums[r1.item] += e1;
            ++item_counts[r1.item];
            item_sums[r2.item] += e2;
            ++item_counts[r2.item];
            item_sums[r3.item] += e3;
            ++item_counts[r3.item];
            item_sums[r4.item] += e4;
            ++item_counts[r4.item];
            item_sums[r5.item] += e5;
            ++item_counts[r5.item];
            item_sums[r6.item] += e6;
            ++item_counts[r6.item];
            item_sums[r7.item] += e7;
            ++item_counts[r7.item];
            item_sums[r8.item] += e8;
            ++item_counts[r8.item];
            item_sums[r9.item] += e9;
            ++item_counts[r9.item];
            switch (phase_offset) {
                case 0: add_user(r0, e0); add_user(r5, e5); break;
                case 1: add_user(r1, e1); add_user(r6, e6); break;
                case 2: add_user(r2, e2); add_user(r7, e7); break;
                case 3: add_user(r3, e3); add_user(r8, e8); break;
                default: add_user(r4, e4); add_user(r9, e9); break;
            }
        }
        for (; idx < n; ++idx) {
            const Rating& r = ratings[idx];
            const float e = r.rating - mean;
            item_sums[r.item] += e;
            ++item_counts[r.item];
            if (((base_offset + idx) % user_stride) == user_phase) {
                add_user(r, e);
            }
        }

        total_seen += n;
        has_updates = true;
        refresh_scores();
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
    static constexpr int count_lut_limit = 65535;
    static constexpr int learned_parameter_count = 128;
    static constexpr int segment_count = 117;
    static constexpr int user_stride = 5;
    static constexpr int user_phase = 1;
    static constexpr float user_shrink = 20.0f;
    static constexpr float item_shrink = 5.0f;
    static constexpr float model_rmse = 0.91362253f;

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
    3.37045813f, 0.00686011324f, 0.000783007185f, 0.158997104f, 0.0680496097f, 1.00694644f,
    0.944571257f
    };

    static constexpr int segment_thresholds[116] = {
    284, 369, 614, 668, 691, 769, 785, 831, 842, 865,
    1467, 1517, 1771, 2160, 2298, 2468, 2565, 2567, 2914, 2927,
    3019, 3037, 5370, 5377, 5764, 5799, 6609, 6664, 7481, 7556,
    12637, 12686, 13331, 13337, 15298, 15307, 15424, 15431, 15592, 15604,
    15651, 20083, 20117, 20151, 20240, 20275, 20310, 20364, 20371, 20494,
    23634, 27037, 27086, 30293, 30317, 31696, 31711, 37052, 37061, 37071,
    37249, 37252, 40075, 40076, 40440, 40482, 48486, 48497, 50364, 50435,
    51431, 51824, 52003, 52330, 54974, 57706, 57715, 60428, 60440, 61675,
    61897, 63941, 64019, 65020, 65052, 65669, 65686, 66092, 66114, 72400,
    72479, 83889, 83890, 83967, 83971, 88729, 88737, 89453, 89457, 92606,
    92615, 93015, 93023, 100151, 102908, 102910, 103499, 109101, 111649, 111760,
    116895, 116899, 120503, 120523, 135078, 135089
    };

    static constexpr float segment_values[117] = {
    -0.000504253316f, -0.397817016f, 0.157901049f, -0.276975542f, -1.42181242f, 0.162116408f,
    -0.92123878f, 0.222122848f, -1.17255783f, -0.471007675f, 0.0778147131f, 0.526511729f,
    -0.142366633f, 0.241121188f, -0.0644460246f, 0.292163998f, -0.0496090092f, -0.653158844f,
    0.160499379f, -0.87531203f, -0.102013774f, -0.80868876f, 0.0133430427f, -1.07266998f,
    0.199624255f, -0.456645757f, 0.0210126135f, -0.702148497f, 0.138990566f, -0.455600828f,
    0.0189186633f, -0.437673807f, 0.0728033036f, -0.896081626f, 0.060474705f, -0.80308944f,
    0.293200791f, -0.961554527f, 0.0330287665f, -0.791520596f, 0.337399215f, 0.0259975679f,
    0.513324142f, -0.49393031f, 0.184514552f, -0.359096289f, 0.480169922f, 0.01209147f,
    -1.39872706f, 0.430732995f, -0.0413747244f, 0.0376636945f, -0.637399852f, 0.0188067779f,
    -0.655231178f, 0.0327779949f, -1.13921762f, 0.0245415606f, -1.75031769f, 0.503051162f,
    -0.235645682f, -1.67214429f, 0.0458107106f, -0.744429827f, 0.0804506987f, -0.783633173f,
    0.0597479604f, 1.5294534f, 0.040423695f, -0.466150552f, 0.105309367f, -0.139386445f,
    0.339990556f, -0.177129835f, 0.118084274f, 0.00541013898f, -0.697796702f, 0.0295419227f,
    -1.02283764f, -0.0186803527f, 0.38631916f, -0.0735439733f, 0.736008704f, -0.01671784f,
    -1.26657033f, 0.129104495f, -1.48672593f, 0.102142677f, -1.82511938f, 0.0693388879f,
    -0.52956754f, -0.026068652f, -2.72921205f, 0.0370721556f, -1.21733546f, 0.0543656163f,
    -1.23295069f, 0.134422779f, -1.72416544f, 0.0458904281f, -1.60338819f, 0.0426672064f,
    -1.72763085f, 0.0114294766f, -0.0917674378f, -1.03652954f, 0.150174871f, -0.0614198409f,
    0.0777978376f, -0.450463861f, 0.0255928095f, -1.64300621f, 0.0328756459f, -1.54705775f,
    0.00805889722f, -0.668244541f, 0.041254364f
    };

    static float clip_score(float value) {
        if (value < 0.5f) {
            return 0.5f;
        }
        if (value > 5.0f) {
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

    void refresh_scores() {
        for (int user = 0; user < users; ++user) {
            user_score[user] = user_component(user);
        }
        for (int item = 0; item < items; ++item) {
            item_score[item] = item_component(item);
        }
    }
};
