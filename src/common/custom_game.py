import numpy as np
import vizdoom as vzd


class CustomGame(vzd.DoomGame):
    def __init__(self, localization_noise=0, pose_noise=0):
        super().__init__()
        self.localization_noise_sd = localization_noise
        self.pose_noise_sd = pose_noise
        self.last_true_location = None
        self.last_noisy_location = None

    def new_episode(self):
        super().new_episode()
        self.last_true_location = None
        self.last_noisy_location = None

    def get_agent_location(self):
        # Get true location of agent
        true_x = self.get_game_variable(vzd.GameVariable.POSITION_X)
        true_y = self.get_game_variable(vzd.GameVariable.POSITION_Y)
        true_angle = self.get_game_variable(vzd.GameVariable.ANGLE)

        # Return true location if first time called
        if self.last_true_location is None:
            self.last_true_location = (true_x, true_y, true_angle)
            self.last_noisy_location = (true_x, true_y, true_angle)
            return true_x, true_y, true_angle

        # Get change in location and angle
        (last_true_x, last_true_y, last_true_angle) = self.last_true_location
        diff_x = true_x - last_true_x
        diff_y = true_y - last_true_y
        diff_angle = true_angle - last_true_angle

        # Generate localization noise for agent position
        noise_localization = np.random.normal(1, self.localization_noise_sd)
        noise_pose = np.random.normal(1, self.pose_noise_sd)

        # Simulate agent position obtained by simulated noisy sensor
        (last_x, last_y, last_angle) = self.last_noisy_location
        agent_x = last_x + (noise_localization * diff_x)
        agent_y = last_y + (noise_localization * diff_y)
        agent_angle = last_angle + (noise_pose * diff_angle)

        self.last_true_location = (true_x, true_y, true_angle)
        self.last_noisy_location = (agent_x, agent_y, agent_angle)
        return agent_x, agent_y, agent_angle
