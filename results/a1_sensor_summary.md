# Component A1 - Perception pipeline metrics

| Metric | Value |
|---|---|
| LiDAR beams / hits | 360 / 283 |
| LiDAR noise std | 0.02 m |
| Camera resolution | 600x600 px (60 px/m) |
| Obstacle contours (OpenCV) | 2 |
| Landmarks detected | 1 / 3 |
| LiDAR updates fused | 1 |
| Camera updates fused | 2 |
| Occupancy precision | 0.973 |
| Occupancy recall | 0.993 |

Fusion: log-odds occupancy grid; camera evidence (|L|=1.2) outweighs a
single LiDAR hit (|L|=0.85) because colour classification is noise-free up
to projection quantisation, while LiDAR carries 2 cm Gaussian range noise.
