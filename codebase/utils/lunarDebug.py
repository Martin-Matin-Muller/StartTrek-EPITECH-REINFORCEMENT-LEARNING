import gymnasium as gym
import pygame


class LunarLanderDebug:
    def __init__(self, env):
        self.env = env
        self.total_reward = 0
        self.episode = 0
        self.widgets = {}  # nom -> fonction

        pygame.init()
        self.font = pygame.font.SysFont("Arial", 18)

    # -------------------------
    # API SIMPLE : .add()
    # -------------------------
    def add(self, name, func):
        """
        name  : str
        func  : lambda (env_wrapper, reward) -> value
        """
        self.widgets[name] = func
        return self  # chaining possible

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.total_reward += reward
        self._render_overlay(reward)

        return obs, reward, terminated, truncated, info

    # -------------------------
    # RESET
    # -------------------------
    def reset(self, **kwargs):
        self.total_reward = 0
        self.episode += 1
        return self.env.reset(**kwargs)

    # -------------------------
    # OVERLAY
    # -------------------------
    def _render_overlay(self, reward):
        screen = pygame.display.get_surface()
        if screen is None:
            return

        y = 10

        for name, func in self.widgets.items():
            try:
                value = func(self, reward)
                text = self.font.render(f"{name}: {value:.2f}", True, (255, 255, 255))
                screen.blit(text, (10, y))
                y += 22
            except:
                pass

        pygame.display.flip()

# Debug
# env = LunarLanderDebug(env)
# env.add("Total", lambda e, r: e.total_reward)
# env.add("Episode", lambda e, r: e.episode)

# observation, info = env.reset(seed=42)

# for _ in range(1000):
#     action = env.env.action_space.sample()

#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.env.close()
