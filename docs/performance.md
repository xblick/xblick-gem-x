# Accuracy and speed

This page reports the one test of durchblick-gem-x so far. The test is synthetic, because we had no recording of
people with depth.

## Setup

Measured on 2026-11-01 on an RTX PRO 6000.

| Part | Value |
| --- | --- |
| Colour | GEM-X's demo video, 640 x 360, two persons, published as a camera through Durchblick |
| Depth | rendered from GEM-X's own meshes of the two persons, moved 0.30 m farther away |
| Rendering of the depth | dense, from triangles. The node's depth fit uses the vertices only. |
| Noise and background | 1 cm noise, the background at 6 m |
| Length | 40 s at 10 frames per second, 796 bodies |

The depth is 0.30 m farther than the bodies that GEM-X estimates, so the true bodies are known. The frames of the
demo video already contain NVIDIA's rendering of GEM-X's mesh, and the node sees this rendering as the person.

## Accuracy

| Measure | Result |
| --- | --- |
| Shift of the depth fit | median 0.296 m, 5 to 95 %: 0.292 to 0.299 m. The true shift is 0.30 m. |
| Bodies with a fit | all 796 |
| Joints against the true joints | median 4.0 mm, 95 %: 8.1 mm |
| Ids | both persons kept their id over the whole recording |

## Speed

| Step | Time |
| --- | --- |
| Whole frame with two persons | 59 to 60 ms |
| YOLOX on the GPU | 16 ms |
| YOLOX on the CPU (`--detector-device cpu`) | 200 to 450 ms |
| GEM-X per person in bfloat16 | 30 ms |
| GEM-X per person in float32 (`--autocast none`) | 43 ms |

The node keeps up with the 10 frames per second of the camera and has time to spare. The joints in bfloat16 are
within 2 mm of the joints in float32.

The node alone uses 7.5 GiB of VRAM with one person in view (RTX PRO 6000, 1280 x 720 at 30 frames per second,
measured 2026-11-02).

## Status and limits

- The test checks that the parts of the node work together, and it checks the depth fit. It does not measure the
  accuracy of GEM-X.
- GEM-X's README states the errors of Body Pose: 62 mm MPJPE on 3DPW and 68 mm on EMDB. MPJPE is the mean distance
  between estimated and true joints.
- We have not measured how much the depth fit improves these errors on real people. This needs a camera with people
  in front of it.
- We have not compared the node with the Azure Kinect tracker.
