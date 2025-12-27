Enums
=====

Modality
--------

Enumeration of data modalities available in StandardE2E datasets.

.. autoclass:: standard_e2e.enums.Modality
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value

   **Available Modalities:**

   .. attribute:: CAMERAS
      
      Multi-view camera images with intrinsics and extrinsics

   .. attribute:: LIDAR_BEV
      
      LiDAR bird's eye view representation

   .. attribute:: LIDAR_PC
      
      LiDAR point cloud data

   .. attribute:: HD_MAP
      
      High-definition map elements

   .. attribute:: SPEED
      
      Vehicle speed information

   .. attribute:: INTENT
      
      Driver intent (go straight, turn left/right)

   .. attribute:: FUTURE_STATES
      
      Future trajectory states

   .. attribute:: PAST_STATES
      
      Past trajectory states

   .. attribute:: PREFERENCE_TRAJECTORY
      
      Preferred or expert trajectory

   .. attribute:: DETECTIONS_3D
      
      3D object detections

Intent
------

Enumeration of driver intent values.

.. autoclass:: standard_e2e.enums.Intent
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value

   **Intent Values:**

   .. attribute:: UNKNOWN
      :annotation: = 0
      
      Unknown or unspecified intent

   .. attribute:: GO_STRAIGHT
      :annotation: = 1
      
      Intent to continue straight

   .. attribute:: GO_LEFT
      :annotation: = 2
      
      Intent to turn left

   .. attribute:: GO_RIGHT
      :annotation: = 3
      
      Intent to turn right

CameraDirection
---------------

Enumeration of camera mounting directions.

.. autoclass:: standard_e2e.enums.CameraDirection
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value

   **Camera Directions:**

   .. attribute:: FRONT
      
      Front-facing camera

   .. attribute:: FRONT_LEFT
      
      Front-left camera

   .. attribute:: FRONT_RIGHT
      
      Front-right camera

   .. attribute:: SIDE_LEFT
      
      Left side camera

   .. attribute:: SIDE_RIGHT
      
      Right side camera

   .. attribute:: REAR
      
      Rear-facing camera

   .. attribute:: REAR_LEFT
      
      Rear-left camera

   .. attribute:: REAR_RIGHT
      
      Rear-right camera

TrajectoryComponent
-------------------

Enumeration of trajectory data components.

.. autoclass:: standard_e2e.enums.TrajectoryComponent
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value

   **Trajectory Components:**

   .. attribute:: TIMESTAMP
      
      Timestamp of the trajectory point

   .. attribute:: IS_VALID
      
      Validity flag for the trajectory point

   .. attribute:: X
      
      X coordinate

   .. attribute:: Y
      
      Y coordinate

   .. attribute:: Z
      
      Z coordinate

   .. attribute:: VX
      
      Velocity in X direction

   .. attribute:: VY
      
      Velocity in Y direction

   .. attribute:: HEADING
      
      Heading angle

   .. attribute:: WIDTH
      
      Object width

   .. attribute:: LENGTH
      
      Object length

DetectionType
-------------

Enumeration of object detection types.

.. autoclass:: standard_e2e.enums.DetectionType
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value
