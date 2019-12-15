import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.init_pose = self.sim.pose[:3]
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0
        penalty = 0

        # weights for distance axis. For take-off, the
        # y-axis is weighted the most relevant.
        weights = np.array([0.8, 0.8, 1.])

        # additional incentive for proximity milestones.
        # if within these boundaries, the machine will be
        # encouraged greatly
        proximity_incentive = {
            10: 100,
            1: 500,
        }

        # Compute stability as a factor (by euler angles)
        # These do not have weights as tipping in any of the directions is necessary for movement
        #
        # Large offsets should be avoided as we risk moving uncontrollably or just flipping (see `try_your_best.png`)
        euler = abs(self.sim.pose[3:6]).sum()

        # Compute distance to target from current position
        dist_height = self.sim.pose[2] - self.target_pos[2]
        distance = np.sqrt(np.power(abs(self.sim.pose[:3] - self.target_pos), 2).sum())
        weighted_dist = np.sqrt((weights * np.power(abs(self.sim.pose[:3] - self.target_pos), 2)).sum())

        # Penalty for distance from target and stability issues (disabled for now - see `try_your_best.png`)
        # penalty = euler + weighted_dist
        penalty = weighted_dist

        # Proximity based 'bonus' incentives when we reach thresholds for the targets
        # for dist in proximity_incentive:
        #     if distance < dist:
        #         reward += proximity_incentive[dist]

        weight = .003 # 3x10^-3

        # Use tanh to restrict the reward between [-1,1].
        return np.tanh(1. - weight * abs(reward - penalty))

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state