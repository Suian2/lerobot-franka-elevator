# Franka ACT Rollout Action Queue Fix

## Problem

`hardware_test/franka/run_act_rollout.py` calls `predict_action_chunk()` on every
control tick and always executes element zero. The trained ACT policy returns a
100-step trajectory whose first samples are intentionally almost stationary;
discarding steps 1–99 therefore makes a successful rollout look motionless.
INFO logging is also hidden because `logging.basicConfig()` does not replace a
WARNING-level handler installed during imports.

## Decision

Use ACT's public `select_action()` API. It owns the policy action queue and
executes the predicted trajectory in order, matching LeRobot's ACT tutorial and
synchronous rollout implementation. Reset policy and processor inference state
once before each rollout. Keep the first selected action as the first executed
action instead of treating it as a discarded warm-up.

The alternative of manually caching chunks is rejected because it duplicates
ACT queue semantics. Periodic receding-horizon replanning is deferred because it
introduces an unvalidated control parameter.

## Safety and Observability

Do not change the existing hard limits: 30 Hz, at most 20 seconds,
`action_scale <= 0.25`, linear velocity at most 0.05 m/s, angular velocity at
most 0.20 rad/s, gripper suppressed, and zero Cartesian velocity before and
after execution.

Force the script's INFO logging configuration so the operator sees DRY_RUN vs
EXECUTE, the first scaled action, sent frame count, elapsed time, and achieved
loop rate. No per-frame logging is added because it would disturb timing.

## Tests

- A fake queued ACT policy returns distinct sequential actions; the control loop
  must send them in order instead of repeatedly sending chunk step zero.
- The first preselected action must be executed, stop requests must prevent the
  next send, and zero velocity must still be sent before and after execution.
- The updated CLI safety bounds remain enforced and dry-run behavior remains
  non-executing.
- Logging setup must explicitly replace the pre-existing root handler.
