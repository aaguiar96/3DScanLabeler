#!/usr/bin/env python

import rospy
import sys
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

class ScanLabeler:
    # Class constructor
    # - get topic names and bringup callback functions
    def __init__(self):
        # ROS subscribers
        self.sub   = rospy.Subscriber("velodyne_points", PointCloud2, self.scanCallback)

        # Plane fitting parameters
        self.A    = 0
        self.B    = 0
        self.C    = 0
        self.D    = 0
        self.refA = 0
        self.refB = 0
        self.refC = 0
        self.refD = 0
        n_inliers = 0

        seed_point = [1.78, 0.62, -0.36]

    # Callback function to read 3D scan measures
    def scanCallback(self, data):
        # Extract 3D point from the LiDAR scan
        pts = pc2.read_points(data, skip_nans = True, field_names = ("x", "y", "z"))
        # Filter points by distance to the seed point (considering 1.5 meters)
        #for p in pts:
            #print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])

    # Defines a plane using a set of points and least squares minimization
    def fitPlaneLTSQ(XYZ):
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
        # Ransac iterations
        for i in range(0, number_iterations):

            # Randomly select three points that connot be cohincident
            # TODO missing check: the points also cannot be colinear
            idx1 = random.randint(0, number_points)
            while True:
                idx2 = random.randint(0, number_points)
                if not idx2 == idx1:
                    break
            while True:
                idx3 = random.randint(0, number_points)
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
            num_inliers = (distances < distance_threshold).sum()

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
        inliers = pts[np.where(distances < distance_threshold)]

        # Refine the plane model by fitting a plane to all inliers
        c, normal = fitPlaneLTSQ(inliers)
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
    scan_handler = ScanLabeler()
    rospy.init_node("scan_labeler")

    # ROS spin until user break it
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
