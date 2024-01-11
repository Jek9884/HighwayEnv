from typing import Dict, Text

import numpy as np


from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle


class LaneCenteringEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "TimeToCollision",
                "horizon": 16
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [8, 16, 24]
            },
            "screen_width": 600,
            "screen_height": 600,
            "duration": 100,
            "collision_reward": -1.0,  # Penalization received for vehicle collision.
            "left_lane_reward": 0.,  # Reward received for maintaining left most lane.
            "high_speed_reward": 10,  # Reward received for maintaining cruising speed.
            "lane_centering_cost": 0.5,
            "lane_centering_reward": 1,
            "reward_speed_range": [10, 50],
            "normalize_reward": True,
            "offroad_terminal": True,
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
                                         self.config["left_lane_reward"] +
                                         self.config["lane_centering_reward"]], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.lane_index[2]
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

    def _make_road(self, length=128):
        rng = self.np_random
        net = RoadNetwork()
        amplitude = rng.uniform(1,20, 1)
        lane = SineLane([0, StraightLane.DEFAULT_WIDTH], 
                        [500, StraightLane.DEFAULT_WIDTH], 
                        amplitude=amplitude[0], 
                        pulsation=2*np.pi / 100, 
                        phase=0,
                        width=10,
                        line_types=[LineType.STRIPED, LineType.STRIPED])
        net.add_lane("a", "b", lane)
        # other_lane = StraightLane([50, StraightLane.DEFAULT_WIDTH], [115, StraightLane.DEFAULT_WIDTH],
        #                           line_types=(LineType.STRIPED, LineType.STRIPED))
        # net.add_lane("c", "d", other_lane)
        # self.lanes = [other_lane, lane]
        # self.lane = self.lanes.pop(0)
        # net.add_lane("d", "a", StraightLane([115, 15], [115+20, 15+20*(15-50)/(115-50)],
        #                                     line_types=(LineType.NONE, LineType.STRIPED), width=10))
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Strategic addition of vehicles for testing safety behavior limits
        while performing U-Turn manoeuvre at given cruising interval.

        :return: the ego-vehicle
        """

        ego_lane = self.road.network.get_lane(("a", "b", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(0, 0),
                                                     speed=16)

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
