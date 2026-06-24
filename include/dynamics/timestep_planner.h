#pragma once

#include <vector>

#include "core/body.h"
#include "dynamics/pairing.h"

struct PairTimestepInfo {
    Pair pair;
    double timescale;
    double suggested_dt;
    int level;
};
struct TimestepPlan {
    bool enabled;
    double base_dt;
    int max_level;
    std::vector<PairTimestepInfo> pair_info;
};
struct TimestepLevelSchedule {
    int level;
    double dt;
    int substeps_per_base_step;
    std::vector<Pair> pairs;
};
struct TimestepSchedule {
    bool enabled;
    double base_dt;
    int max_level;
    std::vector<TimestepLevelSchedule> levels;
};
struct HernandezPairLevelGroup {
    int level = 0;
    double dt = 0.0;
    std::vector<Pair> pairs;
};
struct HernandezPairLevelSchedule {
    double base_dt = 0.0;
    int max_level = 0;
    std::vector<HernandezPairLevelGroup> levels;
};
struct AdaptiveLevelState {
    int active_level = 0;
    int pending_lower_level = -1;
    int pending_lower_level_count = 0;
};

TimestepPlan build_timestep_plan(const std::vector<Body>& bodies, const std::vector<Pair>& pairs, double base_dt, double G, int max_level, double eta);
TimestepSchedule build_timestep_schedule(const TimestepPlan& plan);
HernandezPairLevelSchedule build_hernandez_pair_level_schedule(const TimestepPlan& plan);
HernandezPairLevelSchedule restrict_hernandez_pair_level_schedule(const HernandezPairLevelSchedule& schedule, int active_level);

int deepest_nonempty_hernandez_level(const HernandezPairLevelSchedule& schedule);

void update_adaptive_level_state(AdaptiveLevelState& state, int planner_level, int decrease_delay, bool first_refresh);
void print_timestep_plan_summary(const TimestepPlan& plan);
void print_timestep_schedule_summary(const TimestepSchedule& schedule);
void print_hernandez_pair_level_schedule(const HernandezPairLevelSchedule& schedule);