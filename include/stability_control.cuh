#pragma once

#include <algorithm>
#include <cmath>

#include "grid.cuh"

namespace gpuwm {

struct StabilityControlConfig {
    int enabled = 1;
    double target_divergence_cfl = 0.05;
    double target_divergence_ratio = 0.35;
    double max_diffusion_multiplier = 6.0;
    double base_pressure_retain = 0.90;
    double min_pressure_retain = 0.65;
    int w_cfl_damping = 0;
    double w_damping_alpha = 0.3;
    double w_damping_beta = 1.0;
    double w_transport_blend = 1.0;
    int w_transport_diagnostics = 0;
    int disable_slow_w_metric = 0;
};

struct FlowControlMetrics {
    double mean_abs_div = 0.0;
    double max_abs_div = 0.0;
    double mean_div = 0.0;
    double mean_hdiv = 0.0;
    double mean_dwdz = 0.0;
    double max_abs_hdiv = 0.0;
    double max_abs_dwdz = 0.0;
    double mean_abs_vort = 0.0;
    double max_abs_vort = 0.0;
    double mean_vort2 = 0.0;
};

struct AdaptiveStabilityState {
    double kdiff_scale = 1.0;
    double pressure_retain = 0.90;
    double div_cfl = 0.0;
    double vort_cfl = 0.0;
    double div_ratio = 0.0;
    double control_signal = 0.0;
};

struct WTransportDiagnostics {
    double mean_abs_old_total = 0.0;
    double mean_abs_new_total = 0.0;
    double mean_abs_delta = 0.0;
    double mean_delta = 0.0;
    double mean_divergence = 0.0;
    double rms_delta = 0.0;
    double rms_divergence = 0.0;
    double delta_div_correlation = 0.0;
    double samples = 0.0;
    double tendency_calls = 0.0;
};

inline AdaptiveStabilityState evaluate_adaptive_stability(
    const StabilityControlConfig& cfg,
    const FlowControlMetrics& metrics,
    double dt
) {
    AdaptiveStabilityState state;
    state.pressure_retain = cfg.base_pressure_retain;

    state.div_cfl = metrics.mean_abs_div * dt;
    state.vort_cfl = metrics.mean_abs_vort * dt;
    state.div_ratio = metrics.mean_abs_div / std::max(metrics.mean_abs_vort, 1.0e-8);

    if (!cfg.enabled) {
        return state;
    }

    double div_excess = std::max(state.div_cfl / std::max(cfg.target_divergence_cfl, 1.0e-8) - 1.0, 0.0);
    double ratio_excess = std::max(state.div_ratio / std::max(cfg.target_divergence_ratio, 1.0e-8) - 1.0, 0.0);
    double peak_div_excess = std::max(metrics.max_abs_div * dt /
        std::max(2.0 * cfg.target_divergence_cfl, 1.0e-8) - 1.0, 0.0);

    state.control_signal = 0.6 * div_excess + 0.3 * ratio_excess + 0.1 * peak_div_excess;
    state.kdiff_scale = std::min(1.0 + state.control_signal, cfg.max_diffusion_multiplier);
    state.pressure_retain = std::max(
        cfg.min_pressure_retain,
        cfg.base_pressure_retain - 0.08 * state.control_signal
    );

    return state;
}

FlowControlMetrics compute_flow_control_metrics(const StateGPU& state, const GridConfig& grid);
WTransportDiagnostics consume_w_transport_diagnostics();
void reset_w_transport_diagnostics();

} // namespace gpuwm
