# ROS2 image bridge reliable QoS design

## Context

The L515 publishes `/l515/color/image_raw` at 30 Hz according to its own
diagnostics. The Python ZMQ bridge receives only 13-24 Hz with large gaps.
Controlled subscriptions showed about 27.8 Hz with RELIABLE delivery and
20.6 Hz with BEST_EFFORT delivery while keeping history, depth, and durability
equivalent.

## Decision

Change only the bridge subscription reliability from BEST_EFFORT to RELIABLE.
Keep `KEEP_LAST`, depth 1, VOLATILE durability, the 960x540 camera profile, and
the existing single-frame ZMQ protocol unchanged.

This is preferred over changing the camera publisher QoS because RELIABLE has
already been measured against the live publisher. It is preferred over reducing
resolution because a resolution change could shift the ACT policy's visual
input distribution.

## Rollback

Archive the current untracked bridge and its existing tests in a dedicated Git
baseline commit before applying the change. Apply the QoS change in a separate
commit so either `git revert` or restoration from the baseline commit is
possible without touching unrelated worktree changes.

## Testing

1. Add a behavioral unit test that runs bridge setup with fake ROS2/ZMQ modules
   and observes the QoS passed to `create_subscription`.
2. Verify the test fails because the current reliability is BEST_EFFORT.
3. Change the production reliability to RELIABLE and verify the focused tests.
4. Run Ruff on the changed Python files.
5. With the existing L515 launch left running, measure the bridge using a
   5-second statistics window. Keep the change only if it improves over the
   observed 13-24 Hz range without introducing camera errors; otherwise restore
   the baseline.

## Scope

No robot commands, policy inference, camera restart, dataset changes, or model
changes are included.
