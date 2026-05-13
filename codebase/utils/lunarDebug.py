import gymnasium as gym
import pygame


class LunarLanderDebug:
    def __init__(self, env):
        self.env = env
        self.total_reward = 0
        self.episode = 0
        self.widgets = {}   # nom -> func(wrapper, reward) -> value
        self._state = {}   # nom -> valeur pushée via .update()

        pygame.init()
        self.font = pygame.font.SysFont("Arial", 18)

    # -------------------------
    # API SIMPLE : .add()
    # -------------------------
    def add(self, name, func=None):
        """
        name : str
        func : lambda (env_wrapper, reward) -> value   (optionnel)
        Si func est None, la valeur doit être fournie via .update()
        """
        self.widgets[name] = func
        return self

    # -------------------------
    # UPDATE  ← nouveau
    # -------------------------
    def update(self, name, value):
        """
        Pousse une valeur externe pour le widget `name`.
        Prioritaire sur la lambda éventuelle.

        Usage :
            env.add("Fuel")          # sans lambda
            env.update("Fuel", fuel) # appelé à chaque step
        """
        self._state[name] = value
        return self

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
        self._state.clear()          # remet les valeurs à zéro entre épisodes
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
                # Priorité : valeur pushée > résultat de la lambda
                if name in self._state:
                    value = self._state[name]
                elif func is not None:
                    value = func(self, reward)
                else:
                    continue

                text = self.font.render(f"{name}: {value:.2f}", True, (255, 255, 255))
                screen.blit(text, (10, y))
                y += 22
            except Exception:
                pass

        pygame.display.flip()