from typing import Dict, Text

import numpy as np


from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle


class IndianaEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
            },
            "action": {
                "type": "Continuous",
            },
            "screen_width": 600,
            "screen_height": 600,
            "duration": 300,
            "collision_reward": -1.0,  # Penalization received for vehicle collision.
            "left_lane_reward": 0.,  # Reward received for maintaining left most lane.
            "high_speed_reward": 10,  # Reward received for maintaining cruising speed.
            "lane_centering_cost": 0.5,
            "lane_centering_reward": 1,
            "reward_speed_range": [10, 50],
            "normalize_reward": True,
            "offroad_terminal": True,
            "controlled_vehicles": 1,
            "initial_speed": 30
        })
        return config

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed and collision avoidance.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        if not self.vehicle.on_road or self.vehicle.crashed:
            return -1
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"],
                                         self.config["high_speed_reward"] + 
                                         self.config["lane_centering_reward"]], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        # neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        # lane = self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        longitudinal, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        if self.config["lane_centering_reward"]:
            lane_centering_rew = 1/(1+self.config["lane_centering_cost"]*lateral**2)
        else:
            lane_centering_rew = 0
        return {
            "collision_reward": self.vehicle.crashed,
            # "left_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "lane_centering_reward": lane_centering_rew,
            "on_road_reward": self.vehicle.on_road
        }

    def _is_terminated(self) -> bool:
        return (self.vehicle.crashed or 
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> np.ndarray:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=80):
        """
        Making double lane road with counter-clockwise U-Turn.
        :return: the road
        """
        net = RoadNetwork()

        # Defining upper starting lanes after the U-Turn.
        # These Lanes are defined from x-coordinate 'length' to 0.
        net.add_lane("c", "d", StraightLane([length, StraightLane.DEFAULT_WIDTH], [0, StraightLane.DEFAULT_WIDTH],
                                            line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED)))
        net.add_lane("c", "d", StraightLane([length, 0], [0, 0],
                                            line_types=(LineType.NONE, LineType.CONTINUOUS_LINE)))

        # Defining counter-clockwise circular U-Turn lanes.
        center = [length, StraightLane.DEFAULT_WIDTH + 20]  # [m]
        center2 = [0, -StraightLane.DEFAULT_WIDTH + 28]  # [m]
        radius = 20  # [m]
        alpha = 0  # [deg]

        radii = [radius, radius+StraightLane.DEFAULT_WIDTH]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("b", "c",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(-90+alpha),
                                      clockwise=False, line_types=line[lane]))
        # the U-Turn lanes on left 
        net.add_lane("d", "a",
                        CircularLane(center2, radii[1], np.deg2rad(270 - alpha), np.deg2rad(90+alpha),
                                    clockwise=False, line_types=line[1]))
        net.add_lane("d", "a",
                        CircularLane(center2, radii[0], np.deg2rad(270 - alpha), np.deg2rad(90+alpha),
                                    clockwise=False, line_types=line[0]))
        
        offset = 2*radius

        # Defining lower starting lanes before the U-Turn.
        # These Lanes are defined from x-coordinate 0 to 'length'.
        net.add_lane("a", "b", StraightLane([0, ((2 * StraightLane.DEFAULT_WIDTH + offset) - StraightLane.DEFAULT_WIDTH)],
                                            [length, ((2 * StraightLane.DEFAULT_WIDTH + offset) - StraightLane.DEFAULT_WIDTH)],
                                            line_types=(LineType.CONTINUOUS_LINE,
                                                        LineType.STRIPED)))
        
        net.add_lane("a", "b", StraightLane([0, (2 * StraightLane.DEFAULT_WIDTH + offset)],
                                            [length, (2 * StraightLane.DEFAULT_WIDTH + offset)],
                                            line_types=(LineType.NONE,
                                                        LineType.CONTINUOUS_LINE)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            # lane_index = self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=self.config['initial_speed'],
                                                                             longitudinal=rng.uniform(20, 20)
                                                                             )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=30+rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["vehicles_count"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=30+rng.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)
