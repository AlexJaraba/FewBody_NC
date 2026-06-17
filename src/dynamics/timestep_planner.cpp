#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "dynamics/timestep_planner.h"
#include "math/vec3.h"

/* ==========================================================================================

    Adaptive timestep planner

    This file estimates hierarchy-aware timestep levels for body pairs.

    Level convention:
        level = 0 uses the global base timestep dt
        level = 1 suggests dt / 2
        level = 2 suggests dt / 4
        ...

   ========================================================================================== */

namespace {
    double safe_pair_timescale(const Body& a, const Body& b, double G) {
        const Vec3 dr = b.position - a.position;
        const Vec3 dv = b.velocity - a.velocity;

        const double r = dr.norm();
        const double v = dv.norm();
        const double mass_sum = a.mass + b.mass;
        const double tiny = 1e-300;
        const double safe_r = std::max(r, tiny);
        const double safe_mass = std::max(mass_sum, tiny);

        double t_dyn = std::numeric_limits<double>::infinity();
        double t_cross = std::numeric_limits<double>::infinity();

        if (G > 0.0 && safe_mass > tiny) {
            t_dyn = std::sqrt((safe_r * safe_r * safe_r) / (G * safe_mass));
        }
        if (v > tiny) {
            t_cross = safe_r / v;
        }

        const double timescale = std::min(t_dyn, t_cross);

        if (!std::isfinite(timescale) || timescale <= 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        return timescale;
    }

    int choose_level(double base_dt, double suggested_dt, int max_level) {
        if (max_level <= 0) {
            return 0;
        }
        if (!std::isfinite(suggested_dt) || suggested_dt <= 0.0) {
            return max_level;
        }

        int level = 0;
        double candidate_dt = base_dt;

        while (level < max_level && candidate_dt > suggested_dt) {
            candidate_dt *= 0.5;
            ++level;
        }
        return level;
    }
};

TimestepPlan build_timestep_plan(const std::vector<Body>& bodies, const std::vector<Pair>& pairs, double base_dt, double G, int max_level, double eta) {
    TimestepPlan plan;
    plan.enabled = max_level > 0;
    plan.base_dt = base_dt;
    plan.max_level = std::max(0, max_level);

    const double safe_eta = std::max(eta, 1e-12);

    plan.pair_info.reserve(pairs.size());

    for (const Pair& pair : pairs) {
        if (pair.i < 0 || pair.j < 0) {
            continue;
        }
        if (pair.i >= static_cast<int>(bodies.size()) || pair.j >= static_cast<int>(bodies.size())) {
            continue;
        }

        const Body& a = bodies[pair.i];
        const Body& b = bodies[pair.j];

        const double timescale = safe_pair_timescale(a, b, G);
        const double suggested_dt = safe_eta * timescale;

        const int level = choose_level(base_dt, suggested_dt, plan.max_level);

        plan.pair_info.push_back({pair, timescale, suggested_dt, level});
    }
    return plan;
}

void print_timestep_plan_summary(const TimestepPlan& plan) {
    std::cout << "\n=== Adaptive Timestep Planner Diagnostics ===\n";
    std::cout << "Diagnostic-only mode: integration still uses fixed global dt.\n";
    std::cout << "Base dt: " << plan.base_dt << "\n";
    std::cout << "Max level: " << plan.max_level << "\n";

    if (plan.pair_info.empty()) {
        std::cout << "No pair timestep information available.\n";
        return;
    }

    std::vector<int> counts(plan.max_level + 1, 0);

    for (const PairTimestepInfo& info : plan.pair_info) {
        if (info.level >= 0 && info.level <= plan.max_level) {
            counts[info.level] += 1;
        }
    }
    for (int level = 0; level <= plan.max_level; ++level) {
        double dt_level = plan.base_dt;
        for (int k = 0; k < level; ++k) {
            dt_level *= 0.5;
        }

        std::cout << "Level " << level << " dt = " << dt_level << " pair_count = " << counts[level] << "\n";
    }

    std::cout << "Pair details:\n";

    for (const PairTimestepInfo& info : plan.pair_info) {
        std::cout << " pair (" << info.pair.i << ", " << info.pair.j << ")"
                  << " timescale = " << info.timescale
                  << " suggested_dt = " << info.suggested_dt
                  << " level = " << info.level
                  << "\n";
    }
    std::cout << "=========================================================\n\n";
}