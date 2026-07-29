"""Pygame renderer for rocket attitude state."""

from __future__ import annotations

import math

import numpy as np

from .simulation import SimulationState


class PygameRenderer:
    width = 800
    height = 600

    def __init__(self, display: bool = False) -> None:
        import pygame

        self.pygame = pygame
        pygame.font.init()
        self.display = display
        if display:
            pygame.display.init()
            self.surface = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Rocket Attitude Control")
            self.clock = pygame.time.Clock()
        else:
            self.surface = pygame.Surface((self.width, self.height))
            self.clock = None
        self.font = pygame.font.Font(None, 36)
        self.force_font = pygame.font.Font(None, 27)

    @staticmethod
    def _dashed_line(surface, color, start, end, width=1, dash=10) -> None:
        import pygame

        x1, y1 = start
        x2, y2 = end
        if x1 == x2:
            for y in range(y1, y2, dash * 2):
                pygame.draw.line(surface, color, (x1, y), (x1, min(y + dash, y2)), width)
        elif y1 == y2:
            for x in range(x1, x2, dash * 2):
                pygame.draw.line(surface, color, (x, y1), (min(x + dash, x2), y1), width)

    def _text(
        self,
        value: str,
        position: tuple[int, int],
        color=(245, 245, 220),
        font=None,
    ) -> None:
        active_font = font if font is not None else self.font
        self.surface.blit(active_font.render(value, True, color), position)

    def render(self, state: SimulationState) -> np.ndarray:
        pygame = self.pygame
        white = (245, 245, 220)
        black = (30, 30, 30)
        orange = (255, 165, 0)
        red = (255, 0, 0)
        green = (0, 255, 0)
        yellow = (255, 255, 100)
        maroon = (156, 0, 0)
        orange_yellow = (255, 200, 0)
        self.surface.fill(black)

        self._text(f"Time: {state.time:.2f}", (50, 25), yellow)
        pygame.draw.line(self.surface, yellow, (0, 65), (800, 65), 2)
        for index, force in enumerate(state.thrust):
            self._text(
                f"F{index + 1}: {force:.0f}",
                (24 + index * 96, 105),
                font=self.force_font,
            )
        self._text(f"M_x: {state.moments[0]:.2f}", (50, 150))
        self._text(f"M_y: {state.moments[1]:.1f}", (300, 150))
        self._text(f"M_z: {state.moments[2]:.1f}", (550, 150))
        self._dashed_line(self.surface, yellow, (0, 205), (800, 205), 1, 5)
        self._dashed_line(self.surface, yellow, (0, 430), (800, 430), 1, 5)

        center_x, center_y = 680, 315
        radius = 75
        engine_offset = 15
        theta = np.deg2rad(22.5)
        engines = []
        for index in range(8):
            x = center_x + (-1 if index < 4 else 1) * (radius - engine_offset) * math.sin(theta)
            y = center_y + (1 if index <= 1 or index >= 6 else -1) * (radius - engine_offset) * math.cos(theta)
            engines.append((x, y))
        pygame.draw.circle(self.surface, white, (center_x, center_y), radius, 1)
        self._dashed_line(self.surface, green, (center_x, 230), (center_x, 400), 1, 10)
        self._dashed_line(self.surface, green, (595, center_y), (765, center_y), 1, 10)
        engine_rects = [
            lambda x, y: (x - 5, y, 5, 30),
            lambda x, y: (x - 30, y, 30, 5),
            lambda x, y: (x - 30, y - 5, 30, 5),
            lambda x, y: (x - 5, y - 30, 5, 30),
            lambda x, y: (x, y - 30, 5, 30),
            lambda x, y: (x, y - 5, 30, 5),
            lambda x, y: (x, y, 30, 5),
            lambda x, y: (x, y, 5, 30),
        ]
        for index, (x, y) in enumerate(engines):
            active = state.thrust[index] != 0
            pygame.draw.rect(
                self.surface,
                orange if active else white,
                engine_rects[index](x, y),
                0 if active else 1,
            )

        gamma, psi, phi = state.angles
        phi_surface = pygame.Surface((120, 30), pygame.SRCALPHA)
        pygame.draw.rect(phi_surface, white, (0, 0, 120, 30))
        pygame.draw.rect(phi_surface, red, (90, 0, 30, 10))
        pygame.draw.rect(phi_surface, orange, (90, 0, 4, 8))
        pygame.draw.rect(phi_surface, orange, (90, 22, 4, 8))
        rotated = pygame.transform.rotate(phi_surface, np.rad2deg(phi))
        self.surface.blit(rotated, rotated.get_rect(center=(480, 315)))
        self._dashed_line(self.surface, green, (415, 315), (545, 315), 1, 10)
        pygame.draw.line(self.surface, orange_yellow, (480, 315), (442, 250), 2)
        pygame.draw.rect(self.surface, white, (405, 240, 150, 150), 1)

        psi_surface = pygame.Surface((120, 30), pygame.SRCALPHA)
        pygame.draw.rect(psi_surface, white, (0, 0, 120, 30))
        pygame.draw.rect(psi_surface, red, (90, 0, 30, 30))
        pygame.draw.rect(psi_surface, orange, (90, 0, 4, 8))
        pygame.draw.rect(psi_surface, orange, (90, 22, 4, 8))
        rotated = pygame.transform.rotate(psi_surface, np.rad2deg(psi))
        self.surface.blit(rotated, rotated.get_rect(center=(300, 315)))
        self._dashed_line(self.surface, green, (235, 315), (365, 315), 1, 10)
        pygame.draw.line(self.surface, maroon, (300, 315), (365, 250), 1)
        pygame.draw.line(self.surface, maroon, (300, 315), (365, 380), 1)
        pygame.draw.rect(self.surface, white, (225, 240, 150, 150), 1)

        gamma_surface = pygame.Surface((100, 100), pygame.SRCALPHA)
        pygame.draw.circle(gamma_surface, white, (50, 50), 50)
        pygame.draw.rect(gamma_surface, red, (45, 0, 10, 20))
        rotated = pygame.transform.rotate(gamma_surface, np.rad2deg(-gamma))
        self.surface.blit(rotated, rotated.get_rect(center=(120, 315)))
        self._dashed_line(self.surface, green, (120, 250), (120, 380), 1, 10)
        self._dashed_line(self.surface, green, (55, 315), (185, 315), 1, 10)
        pygame.draw.line(self.surface, maroon, (120, 315), (185, 250), 1)
        pygame.draw.line(self.surface, maroon, (120, 315), (55, 250), 1)
        pygame.draw.rect(self.surface, white, (45, 240, 150, 150), 1)

        rates_deg = np.rad2deg(state.angular_rates)
        angles_deg = np.rad2deg(state.angles)
        for index, name in enumerate(("gamma_dot", "psi_dot", "phi_dot")):
            self._text(f"{name}: {rates_deg[index]:.2f}", (50 + index * 250, 465))
        for index, name in enumerate(("gamma", "psi", "phi")):
            self._text(f"{name}: {angles_deg[index]:.2f}", (50 + index * 250, 525))

        if self.display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise KeyboardInterrupt
            pygame.display.flip()
            assert self.clock is not None
            self.clock.tick(10)
        pixels = pygame.surfarray.array3d(self.surface)
        return np.transpose(pixels, (1, 0, 2)).copy()

    def close(self) -> None:
        if self.display:
            self.pygame.display.quit()
        self.pygame.font.quit()
