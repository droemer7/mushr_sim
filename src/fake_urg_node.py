#!/usr/bin/env python3

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import numpy as np
import range_libc
import rospy
import tf
from geometry_msgs.msg import Quaternion
from nav_msgs.srv import GetMap
from sensor_msgs.msg import LaserScan

import utils


class FakeURGNode:
    def __init__(self):

        self.UPDATE_RATE = float(rospy.get_param("~update_rate"))
        self.THETA_DISCRETIZATION = float(rospy.get_param("~theta_discretization"))
        self.MIN_RANGE_METERS = float(rospy.get_param("~range_min"))
        self.MAX_RANGE_METERS = float(rospy.get_param("~range_max"))
        self.ANGLE_STEP = float(rospy.get_param("~angle_step"))
        self.ANGLE_MIN = float(rospy.get_param("~angle_min"))
        self.ANGLE_MAX = float(rospy.get_param("~angle_max"))
        self.ANGLES = np.arange(
            self.ANGLE_MIN, self.ANGLE_MAX, self.ANGLE_STEP, dtype=np.float32
        )
        self.CAR_LENGTH = float(rospy.get_param("vesc/chassis_length"))
        self.Z_SHORT = float(rospy.get_param("~z_short"))
        self.Z_MAX = float(rospy.get_param("~z_max"))
        self.Z_BLACKOUT_MAX = float(rospy.get_param("~z_blackout_max"))
        self.Z_RAND = float(rospy.get_param("~z_rand"))
        self.Z_HIT = float(rospy.get_param("~z_hit"))
        self.Z_SIGMA = float(rospy.get_param("~z_sigma"))
        self.TF_PREFIX = str(rospy.get_param("~car_name").rstrip("/"))
        self.STANDALONE = bool(rospy.get_param("~standalone"))
        if len(self.TF_PREFIX) > 0:
            self.TF_PREFIX = self.TF_PREFIX + "/"

        map_msg = self.get_map()
        occ_map = range_libc.PyOMap(map_msg)
        max_range_px = round(self.MAX_RANGE_METERS / map_msg.info.resolution)
        self.range_method = range_libc.PyCDDTCast(
            occ_map, max_range_px, self.THETA_DISCRETIZATION
        )
        if not self.STANDALONE:
            self.tl = tf.TransformListener()

            try:
                self.tl.waitForTransform(self.TF_PREFIX + "base_link",
                                         self.TF_PREFIX + "laser_link",
                                         rospy.Time(0),
                                         rospy.Duration(10.0)
                                        )
                position, orientation = self.tl.lookupTransform(
                    self.TF_PREFIX + "base_link",
                    self.TF_PREFIX + "laser_link",
                    rospy.Time(0)
                )
                self.x_offset = position[0]
            except Exception:
                rospy.logwarn("Laser: transform from {0} to {1} not found, "
                              "using no transformation".
                              format(self.TF_PREFIX + "base_link",
                                     self.TF_PREFIX + "laser_link"
                                    )
                             )
                self.x_offset = 0.0
        else:
            self.x_offset = 0.0

        self.laser_pub = rospy.Publisher("laser/scan", LaserScan, queue_size=1)

        self.update_timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.UPDATE_RATE), self.timer_cb
        )

    def noise_laser_scan(self, ranges):
        indices = np.zeros(ranges.shape[0], dtype=np.int)
        prob_sum = self.Z_HIT + self.Z_RAND + self.Z_SHORT
        hit_count = int((self.Z_HIT / prob_sum) * indices.shape[0])
        rand_count = int((self.Z_RAND / prob_sum) * indices.shape[0])
        short_count = indices.shape[0] - hit_count - rand_count
        indices[hit_count : hit_count + rand_count] = 1
        indices[hit_count + rand_count :] = 2
        np.random.shuffle(indices)

        hit_indices = indices == 0
        ranges[hit_indices] += np.random.normal(
            loc=0.0, scale=self.Z_SIGMA, size=hit_count
        )[:]

        rand_indices = indices == 1
        ranges[rand_indices] = np.random.uniform(
            low=self.MIN_RANGE_METERS, high=self.MAX_RANGE_METERS, size=rand_count
        )[:]

        short_indices = indices == 2
        ranges[short_indices] = np.random.uniform(
            low=self.MIN_RANGE_METERS, high=ranges[short_indices], size=short_count
        )[:]

        max_count = (self.Z_MAX / (prob_sum + self.Z_MAX)) * ranges.shape[0]
        while max_count > 0:
            cur = np.random.randint(low=0, high=ranges.shape[0], size=1)
            blackout_count = np.random.randint(low=1, high=self.Z_BLACKOUT_MAX, size=1)
            while (
                cur > 0
                and cur < ranges.shape[0]
                and blackout_count > 0
                and max_count > 0
            ):
                if not np.isnan(ranges[cur]):
                    ranges[cur] = np.nan
                    cur += 1
                    blackout_count -= 1
                    max_count -= 1
                else:
                    break

    def timer_cb(self, event):

        now = rospy.Time.now()
        ls = LaserScan()
        ls.header.frame_id = self.TF_PREFIX + "laser_link"
        ls.header.stamp = now
        ls.angle_increment = self.ANGLE_STEP
        ls.angle_min = self.ANGLE_MIN
        ls.angle_max = self.ANGLE_MAX
        ls.range_min = self.MIN_RANGE_METERS
        ls.range_max = self.MAX_RANGE_METERS
        ls.intensities = []

        ranges = np.zeros(len(self.ANGLES) * 1, dtype=np.float32)

        try:
            base_to_map_trans, base_to_map_rot = self.tl.lookupTransform(
                "/map", self.TF_PREFIX + "base_link", rospy.Time(0)
            )
        except Exception:
            if self.STANDALONE:
                base_to_map_trans = np.zeros(2, dtype=np.float32)
                base_to_map_rot = np.zeros(4, dtype=np.float32)
            else:
                return

        laser_quat = Quaternion()
        laser_quat.x = base_to_map_rot[0]
        laser_quat.y = base_to_map_rot[1]
        laser_quat.z = base_to_map_rot[2]
        laser_quat.w = base_to_map_rot[3]

        laser_angle = utils.quaternion_to_angle(laser_quat)
        laser_pose_x = base_to_map_trans[0] + self.x_offset * np.cos(laser_angle)
        laser_pose_y = base_to_map_trans[1] + self.x_offset * np.sin(laser_angle)

        range_pose = np.array(
            (laser_pose_x, laser_pose_y, laser_angle), dtype=np.float32
        ).reshape(1, 3)
        self.range_method.calc_range_repeat_angles(range_pose, self.ANGLES, ranges)
        ranges = np.clip(ranges, 0.0, self.MAX_RANGE_METERS)
        #self.noise_laser_scan(ranges)
        ls.ranges = ranges.tolist()
        self.laser_pub.publish(ls)

    def get_map(self):
        # Use the 'static_map' service (launched by MapServer.launch) to get the map
        map_service_name = rospy.get_param("~static_map", "/static_map")
        rospy.wait_for_service(map_service_name)
        map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
        return map_msg


if __name__ == "__main__":
    rospy.init_node("fake_urg_node")

    furgn = FakeURGNode()

    rospy.spin()
