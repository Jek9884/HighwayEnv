from itertools import repeat, product
from typing import Tuple, Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle


class RacetrackComplexEnv(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
                "target_speeds": [0, 5, 10]
            },
            "simulation_frequency": 15,
            "policy_frequency": 15,
            "duration": 150,
            "collision_reward": -1,
            "high_speed_reward": 5,
            "lane_centering_cost": 0.2,
            "lane_centering_reward": 1,
            "reward_speed_range": [20, 50],
            # "action_reward": -0.0,
            "controlled_vehicles": 1,
            "vehicles_count": 1,
            "screen_width": 3840,
            "screen_height": 2160,
            "centering_position": [0.5, 0.5],
            "offroad_terminal": True,
            "initial_speed": None,
            "random_spawn":False,
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        if not self.vehicle.on_road or self.vehicle.crashed:
            return -1
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = utils.lmap(reward, [self.config["collision_reward"], self.config["high_speed_reward"]], [0, 1])
        
        #if self.veichle.speed <= 0:
        #    return -reward
        #else:
        #    return reward
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        lane = self.vehicle.lane
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
        lane_heading = lane.heading_at(longitudinal)
        lane_direction = np.array([np.cos(lane_heading), np.sin(lane_heading)])
        forward_velocity = np.dot(self.vehicle.velocity, lane_direction)
        # forward_speed = self.vehicle.speed #* np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_velocity, self.config["reward_speed_range"], [0, 1])
        
        if self.config["lane_centering_reward"]:
            lane_centering_rew = 1/(1+self.config["lane_centering_cost"]*lateral**2)
        else:
            lane_centering_rew = 0
        return {
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "lane_centering_reward": lane_centering_rew,
            # "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            # "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        #net.add_lane("a", "b", lane)
        net.add_lane("a", "b", StraightLane([42-22.15, 6+0.85], [100+141.5, 6+0.85], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=speedlimits[1]))
        net.add_lane("a", "b", StraightLane([42-22.60, 5+6.85], [100+141.5, 5+6.85], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[1]))
        
        """
        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))
        
        # 3 - Vertical Straight
        net.add_lane("c", "d", StraightLane([120, -20], [120, -30],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -20], [125, -30],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2+5, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane("e", "f",
                     CircularLane(center3, radii3+5, np.deg2rad(0), np.deg2rad(136), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 6 - Slant
        net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([59.3934, -19.2], [39.3934, -39.2],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[6]))
        """

        """        # 2 - Circular Arc #2
        center2 = [238.90, -38]
        radii2 = 50
        net.add_lane("v", "w",
                     CircularLane(center2, radii2+4.75, np.deg2rad(-0.25), np.deg2rad(80), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[4]))
        net.add_lane("v", "w",
                     CircularLane(center2, radii2, np.deg2rad(-0.25), np.deg2rad(81), width=5,
                                  clockwise=True, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))"""

        # 5 - Circular Arc #3
        center3 = [234, -42.75]
        radii3 = 50
        net.add_lane("e", "f",
                     CircularLane(center3, radii3+5, np.deg2rad(-0.25), np.deg2rad(85.80), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3, np.deg2rad(-0.25), np.deg2rad(85.80), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 3 - Vertical Straight
        net.add_lane("w", "x", StraightLane([283.95, -40], [283.95, -110],
                                            line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("w", "x", StraightLane([283.95+5, -40], [283.95+5, -110],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))

        # 2 - Circular Arc #2
        center2 = [263.95, -110]
        radii2 = 20
        net.add_lane("x", "y",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("x", "y",
                     CircularLane(center2, radii2+5, np.deg2rad(-0.50), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 3 - Vertical Straight
        net.add_lane("b", "c", StraightLane([239, -73.75], [239, -110],
                                            line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("b", "c", StraightLane([239+5, -73.25], [239+5, -110],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center1 = [218.75, -73.75]
        radii1 = 20
        net.add_lane("c", "d",
                     CircularLane(center1, radii1, np.deg2rad(180), np.deg2rad(0), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("c", "d",
                     CircularLane(center1, radii1+5, np.deg2rad(180), np.deg2rad(0), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # TODO make another curve to go up than a stright way and the final curve
        # 4 - Circular Arc #2
        center2 = [173.75, -73]
        radii2 = 20
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2+5, np.deg2rad(0), np.deg2rad(-182.25), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))


        # 4 - Circular Arc #2
        center1 = [99, -73.75]
        radii1 = 50
        net.add_lane("e", "f",
                     CircularLane(center1, radii1, np.deg2rad(70), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("e", "f",
                     CircularLane(center1, radii1+5, np.deg2rad(70), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 6 - Circular Arc #3
        center3 = [89, -102]
        radii3 = 80
        net.add_lane("f", "g",
                     CircularLane(center3, radii3+5, np.deg2rad(70), np.deg2rad(130), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("f", "g",
                     CircularLane(center3, radii3, np.deg2rad(70), np.deg2rad(130), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane("g", "h",
                     CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                     CircularLane(center4, radii4+5, np.deg2rad(315), np.deg2rad(165), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56+30), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4+5, np.deg2rad(170), np.deg2rad(58+30), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 8 - Circular Arc #5 - Reconnects to Start
        """
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane("i", "a",
                     CircularLane(center5, radii5+5, np.deg2rad(240), np.deg2rad(270), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("i", "a",
                     CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[8]))"""

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            if self.config["random_spawn"]:
                lane_index = self.road.network.random_lane_index(rng)
            else:
                lane_index = ("c", "d", rng.integers(2)) if i == 0 else \
                    self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=self.config["initial_speed"],
                                                                             longitudinal=rng.uniform(20, 20))

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        """vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6+rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["vehicles_count"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6+rng.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)
"""