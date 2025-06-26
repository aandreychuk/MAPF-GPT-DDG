import math
from copy import deepcopy
from itertools import cycle

import numpy as np
from pogema import AnimationConfig, GridConfig
from pogema.svg_animation.animation_drawer import SvgSettings, GridHolder, AnimationDrawer, Drawing
from pogema.wrappers.persistence import AgentState


def create_multi_animation(obstacles, histories: list[list[list[AgentState]]], grid_config: GridConfig,
                           name='render.svg',
                           animation_config: AnimationConfig = AnimationConfig()):
    working_radius = grid_config.obs_radius - 1
    wr = working_radius
    cut_obstacles = np.concatenate([obstacles for _ in range(len(histories))], axis=1)
    cut_obstacles = cut_obstacles[wr:-wr, wr:-wr]

    global_num_agents = sum([len(x) for x in histories])
    history = []
    offset_x = obstacles.shape[1]
    current_offset = 0
    for data in histories:
        history += get_moved_history(data, dy=current_offset, dx=0)
        current_offset += offset_x

    svg_settings = SvgSettings(time_scale=0.4)

    global_idx = 0
    agents_colors = {}
    for num_agents in [len(x) for x in histories]:
        colors_cycle = cycle(svg_settings.colors)
        cur_colors = {index + global_idx: next(colors_cycle) for index in range(num_agents)}
        agents_colors = {**agents_colors, **cur_colors}

        global_idx += num_agents

    episode_sizes = [len(q[0]) for q in histories]
    episode_length = max(episode_sizes)
    for agent_idx in range(global_num_agents):
        while len(history[agent_idx]) <= episode_length:
            q = history[agent_idx][-1]
            inactive = AgentState(q.x, q.y, q.tx, q.ty, q.step, False)
            history[agent_idx].append(inactive)

    grid_holder = GridHolder(
        width=len(cut_obstacles), height=len(cut_obstacles[0]),
        obstacles=cut_obstacles,
        episode_length=episode_length,
        history=history,
        obs_radius=grid_config.obs_radius,
        on_target=grid_config.on_target,
        colors=agents_colors,
        config=animation_config,
        svg_settings=svg_settings
    )

    animation = CustomAnimationDrawer().create_animation(grid_holder)
    with open(name, "w") as f:
        f.write(animation.render())


def get_moved_history(history: list[list[AgentState]], dx=0, dy=0):
    results = []
    for agents in history:
        result_for_agent = []
        for state in agents:
            moved_state = AgentState(state.x + dx, state.y + dy, state.tx + dx, state.ty + dy, state.step, state.active)
            result_for_agent.append(moved_state)
        results.append(result_for_agent)
    return results


def cut_history(history, start, finish):
    history = deepcopy(history)
    for idx, agents_history in enumerate(history):
        history[idx] = agents_history[start:finish]
    return history


class CustomAnimationDrawer(AnimationDrawer):
    def create_animation(self, grid_holder: GridHolder):
        gh = grid_holder
        render_width = gh.height * gh.svg_settings.scale_size + gh.svg_settings.scale_size
        render_height = gh.width * gh.svg_settings.scale_size + gh.svg_settings.scale_size
        drawing = CustomDrawing(width=render_width, height=render_height, svg_settings=SvgSettings())
        obstacles = self.create_obstacles(gh)

        agents = []
        targets = []

        if gh.config.show_agents:
            agents = self.create_agents(gh)
            targets = self.create_targets(gh)

            if not gh.config.static:
                self.animate_agents(agents, gh)
                self.animate_targets(targets, gh)
        if gh.config.show_grid_lines:
            grid_lines = self.create_grid_lines(gh, render_width, render_height)
            for line in grid_lines:
                drawing.add_element(line)
        for obj in [*obstacles, *agents, *targets]:
            drawing.add_element(obj)

        if gh.config.egocentric_idx is not None:
            field_of_view = self.create_field_of_view(grid_holder=gh)
            if not gh.config.static:
                self.animate_obstacles(obstacles=obstacles, grid_holder=gh)
                self.animate_field_of_view(field_of_view, gh)
            drawing.add_element(field_of_view)

        return drawing


class CustomDrawing(Drawing):

    def __init__(self, height, width, svg_settings):
        super().__init__(height, width, svg_settings)

    def render(self):
        scale = max(self.height, self.width) / 1024
        scaled_width = math.ceil(self.width / scale)
        scaled_height = math.ceil(self.height / scale)

        dx, dy = self.origin
        view_box = (dx, dy - self.height, self.width, self.height)

        svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
             width="{scaled_width}" height="{scaled_height}" viewBox="{" ".join(map(str, view_box))}">'''

        definitions = f'''
        <rect id="obstacle" width="{self.svg_settings.r * 2}" height="{self.svg_settings.r * 2}" fill="{self.svg_settings.obstacle_color}" rx="{self.svg_settings.rx}"/>
        <style>
        .line {{stroke: {self.svg_settings.obstacle_color}; stroke-width: {self.svg_settings.stroke_width};}}
        .agent {{r: {self.svg_settings.r};}}
        .target {{fill: none; stroke-width: {self.svg_settings.stroke_width}; r: {self.svg_settings.r};}}
        </style>
        '''

        elements_svg = [svg_header, '<defs>', definitions, '</defs>\n']
        elements_svg.extend(element.render() for element in self.elements)
        elements_svg.append('</svg>')
        return "\n".join(elements_svg)
