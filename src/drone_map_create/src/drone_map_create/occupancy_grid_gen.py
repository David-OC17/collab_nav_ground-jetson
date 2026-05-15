"""
good

now that I have an image of the map I want to post-process it and generate an OccupancyGrid from ROS2

here is a guideline on what I need and a sample image



1. Cut everything outside black + blue stripes and brown contour region (non-map region)

2. Remove perspective/warp

    For horizontal and vertical, independently:

    - Find orientation of blue lines in map

    - Find the average incline

    - Correct to remove perspective

3. Mask areas by color

    - Yellow: obstacles (in-map) --> 75% : occupied

    - Black/blue: free --> 0% : free

    - Brown: obstacles (edge) -->  95% : occupied

    - Other: unknown --> -1 : unknown


A bit more context:

The ground of the map is black, it has blue stripes running vertically and horizontally (they create a grid). There are yellow obstacles (boxes). The map is delimited by a brown (cardboard + wood) barrier. Outside the map most things are white, but could vary in color. There are orange small obstacles (cones) in the map.


The resulting map is only 2D, as it is the plane on which a ground robot operates.

The idea behind detecting the blue lines is to use them as a way to un-warp the resulting map image and make it rectangular/square. The whole map is meant to be square, but it could be that a small portion was not visible during the recording of the drone video.

For now, generate the ROS2 occupancy grid but also save an image of the resulting map after this processing (which would be the grayscale of the occupancy grid) to tune and debug the post processing pipeline.
"""