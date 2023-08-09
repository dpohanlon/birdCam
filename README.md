<p align="center">
  <img width="200" height="200" src="assets/uhohbird.jpg">
</p>

# Bird, interrupted

A simple Flask web app for displaying birds identified on a webcam, with a live feed of the webcam and previous instances of motion detected.

<p align="center">
  <img width="665" height="725" src="assets/interface.png">
</p>

To try and minimise false positives, a `MobileNetV3` model predictions are used to determine whether the object in the frame is a bird or a neighbour out for a 3am stroll in their pyjamas. This works surprisingly well, but seems to be better at identifying bird butts compared to bird faces. More R&D is required.

This web app a single produce multiple consumer interface to the singleton webcam class (using OpenCV), so that motion detection, classification, etc, can be done at the same time as displaying the webcam on the Flask site.
