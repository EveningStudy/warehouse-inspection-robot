# Simulation test

| Measure | Result |
|---|---:|
| Complete missions | 19/20 (95%) |
| Complete patrol routes | 19/20 (95%) |
| Recorded panel approaches | 62/62 |

Three runs were performed one at a time. The remaining 17 used a maximum
concurrency of two simulations. Robot code stayed unchanged throughout all 20 runs.

## Setup

Each run started a fresh simulation on Ubuntu 24.04 in WSL2 with ROS 2 Jazzy.
Gazebo ran headless with its camera sensors active. Each run had a separate
ROS domain and Gazebo partition. The warehouse, route, spawn position and
lighting were the same throughout the test.

Navigation used Gazebo ground-truth odometry and AMCL, with a 0.25 m arrival
tolerance and MPPI GoalCritic weight of 10. RGB and depth frames were paired
within 0.02 seconds; target coordinates used the camera exposure time.
Velocity commands had their own ROS–Gazebo bridge process, started after
robot spawning. Camera and sensor traffic used a separate bridge process.

## Success criteria

A complete mission reaches all 13 patrol waypoints, completes at least one
panel approach and finishes with every approach successful. A complete route
requires all 13 waypoints reached with none skipped.

## Results

All 20 launches count. Nineteen runs reached all 13 waypoints and completed
every recorded approach. No manual motion or recovery commands were sent.
Nav2's built-in recovery remained enabled.

Run 16 timed out after the 30-minute limit. Its spawn process kept waiting
for `/robot_description`, while lifecycle managers waited for Nav2 services.
Patrol never started. This launch remains a failed attempt and was not replaced.
The underlying startup communication issue still needs investigation.

The approach total includes all 62 recorded attempts. The startup failure
produced no approach attempts. Position estimates
can cause repeat visits to the same panel, so physical-panel coverage
needs a separate check. Defect detection and real-world operation remain
outside the tested scope.

The mission success rate was 95%, with a Wilson 95% confidence interval of
76.39%–99.11%. This sample covers one fixed scene under varying machine load.
Other warehouse layouts and real hardware still need testing.

See [individual trials](trials.csv) and the [result summary](summary.json).
