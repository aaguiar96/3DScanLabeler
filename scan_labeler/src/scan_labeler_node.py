#!/usr/bin/env python

import rospy
import std_msgs
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from scipy.spatial import distance

class ScanLabeler:
    # Class constructor
    # - get topic names and bringup callback functions
    def __init__(self):
        # ROS subscribers and publishers
        self.click_sub = rospy.Subscriber("/clicked_point", PointStamped, self.seedCallback)
        self.pcl_sub   = rospy.Subscriber("/velodyne_points", PointCloud2, self.scanCallback)
        self.pcl_pub   = rospy.Publisher("/chessboard_points", PointCloud2, queue_size=1)
        self.seed_pub  = rospy.Publisher("/seed_point", PointCloud2, queue_size=1)
        
        # Plane fitting parameters
        self.A    = 0
        self.B    = 0
        self.C    = 0
        self.D    = 0
        self.refA = 0
        self.refB = 0
        self.refC = 0
        self.refD = 0
        self.n_inliers = 0

        self.init               = 'true'
        self.threshold          = 0.35
        self.distance_threshold = 0.1

    def seedCallback(self, data):
        self.seed_point = np.array([[data.point.x, data.point.y, data.point.z]])
        print(self.seed_point)
        self.init       = 'false'

    # Callback function to read 3D scan measures
    def scanCallback(self, data):
        # Check if seed point was already set
        if self.init == 'true':
            return

        # Extract 3D point from the LiDAR scan
        pc          = ros_numpy.numpify(data)
        points      = np.zeros((pc.shape[0],3))
        points[:,0] = pc['x']
        points[:,1] = pc['y']
        points[:,2] = pc['z']

        # Extract the points close to the seed point from the entire PCL
        dist = distance.cdist(self.seed_point, points, metric='euclidean')
        vec  = points[np.transpose(dist < self.threshold)[:,0], :]

        # Updata seed point with the average of cluster to use in the next 
        # iteration
        if(len(vec) > 0):
            x_sum, y_sum, z_sum = 0, 0, 0
            for i in range(0, len(vec)):
                x_sum += vec[i,0]
                y_sum += vec[i,1]
                z_sum += vec[i,2]
            self.seed_point[0,0] = x_sum / len(vec)
            self.seed_point[0,1] = y_sum / len(vec)
            self.seed_point[0,2] = z_sum / len(vec)

        # Call RANSAC
        inliers = self.ransac(200, vec)

        # Publish plane point into a PCL2
        cloud = pc2.create_cloud_xyz32(data.header, vec)
        self.pcl_pub.publish(cloud)

        # DEBUG - publish seed point
        seed_cloud = pc2.create_cloud_xyz32(data.header, self.seed_point)
        self.seed_pub.publish(seed_cloud)

    # Defines a plane using a set of points and least squares minimization
    def fitPlaneLTSQ(self, XYZ):
        (rows, cols) = XYZ.shape
        G = np.ones((rows, 3))
        G[:, 0] = XYZ[:, 0]  # X
        G[:, 1] = XYZ[:, 1]  # Y
        Z = XYZ[:, 2]
        (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
        normal = (a, b, -1)
        nn = np.linalg.norm(normal)
        normal = normal / nn
        return (c, normal)

    # RANSAC method to extract plane from a PCL
    def ransac(self, number_iterations, pts):
        number_points = pts.shape[0]
        # Ransac iterations
        for i in range(0, number_iterations):

            # Randomly select three points that connot be cohincident
            # TODO missing check: the points also cannot be colinear
            idx1 = random.randint(0, number_points - 1)
            while True:
                idx2 = random.randint(0, number_points - 1)
                if not idx2 == idx1:
                    break
            while True:
                idx3 = random.randint(0, number_points - 1)
                if not idx3 == idx1 and not idx3 == idx2:
                    break

            pt1, pt2, pt3 = pts[[idx1, idx2, idx3], :]

            # ABC Hessian coefficients and given by the external product between two vectors lying on hte plane
            A, B, C = np.cross(pt2 - pt1, pt3 - pt1)
            # Hessian parameter D is computed using one point that lies on the plane
            D = - (A * pt1[0] + B * pt1[1] + C * pt1[2])

            # Compute the distance from all points to the plane
            # from https://www.geeksforgeeks.org/distance-between-a-point-and-a-plane-in-3-d/
            distances = abs((A * pts[:, 0] + B * pts[:, 1] + C * pts[:, 2] + D)) / (math.sqrt(A * A + B * B + C * C))

            # Compute number of inliers for this plane hypothesis.
            # Inliers are points which have distance to the plane less than a distance_threshold
            num_inliers = (distances < self.distance_threshold).sum()

            # Store this as the best hypothesis if the number of inliers is larger than the previous max
            if num_inliers > self.n_inliers:
                self.n_inliers = num_inliers
                self.A         = A
                self.B         = B
                self.C         = C
                self.D         = D

        # Extract the inliers 
        distances = abs((self.A * pts[:, 0] + self.B * pts[:, 1] + self.C * pts[:, 2] + self.D)) / \
                    (math.sqrt(self.A * self.A + self.B * self.B + self.C * self.C))
        inliers   = pts[np.where(distances < self.distance_threshold)]

        # Refine the plane model by fitting a plane to all inliers
        c, normal = self.fitPlaneLTSQ(inliers)
        point = np.array([0.0, 0.0, c])
        d = -point.dot(normal)
        self.refA = normal[0]
        self.refB = normal[1]
        self.refC = normal[2]
        self.refD = d

        # Return the total set of inliers
        return inliers

def main(args):
    # Instantiate class and bringup ROS node
    rospy.init_node("scan_labeler")
    scan_handler = ScanLabeler()

    # ROS spin until user break it
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
