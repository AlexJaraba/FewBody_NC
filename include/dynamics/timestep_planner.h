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

TimestepPlan build_timestep_plan(const std::vector<Body>& bodies, const std::vector<Pair>& pairs, double base_dt, double G, int max_level, double eta);

void print_timestep_plan_summary(const TimestepPlan& plan);