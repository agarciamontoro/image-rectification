# Image rectification

## What is this?

This in an implementation of the Loop&amp;Zhang image rectification algorithm, done as an assigment for the course Computer Vision, at University of Granada.

## How is this done?

 The implementation is made in C++, using OpenCV as a key element of the project. The code follows a functional pattern, mainly implemented in the [util.cpp](https://github.com/agarciamontoro/image-rectification/blob/master/src/util.cpp) file.

## Who has done this?

The algorithm was proposed by Loop&amp;Zhang, while the implementations is done by Antonio Álvarez ([@analca3](https://github.com/analca3/)) and Alejandro García ([@agarciamontoro](https://github.com/agarciamontoro)).

## What does this do?

The code in this repository tries to rectify pairs of stereoschopic images using the Loop&amp;Zhang algorithm, which uses a completely geometric process to obtain the homographies and does not need to know camera-related stuff.

You just need to take two images of the same static scene from two different points of view and the algorithm implemented will rectify them in order to allign its epipolar geometries.

## Can I see some cool results?

Sure! Here you have a pair of Alejandro's cute dog, Lola, whose natural calm is perfect to act as a static scene:

Here you can see the original pair of photos with its epipolar geometry:

![Cute Lola](https://raw.githubusercontent.com/agarciamontoro/image-rectification/master/Informe/Lola-Normal.png)

And the result after apply the algorithm:

![Even cuter Lola](https://raw.githubusercontent.com/agarciamontoro/image-rectification/master/Informe/Lola-Proyectado.png)

Another example, in which the result is quite more impressive, is the following. The original pairs of images from a V for Vendetta toy; the epipolar geometries of the images are quite different:

![Original V](https://raw.githubusercontent.com/agarciamontoro/image-rectification/master/Informe/V-Normal.png)

And the result after have applied the algorithm:

![Rectified V](https://raw.githubusercontent.com/agarciamontoro/image-rectification/master/Informe/V-Proyectado.png)
