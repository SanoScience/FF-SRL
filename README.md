This repository contains the implemenation of the project "FF-SRL: High Performance GPU-Based Surgical Simulation For Robot Learning"

[![arXiv](https://img.shields.io/badge/arXiv-2503.18616-b31b1b.svg)](https://arxiv.org/abs/2503.18616)

In this paper, we presented FF-SRL, a GPU-based simulation environment for robotic surgery that leverages an
advanced XPBD simulation of deformable tissue. We showed that FF-SRL can significantly speed up the RL training
process for surgical tasks, achieving higher frame-rates and faster training time than other available simulators. We also
demonstrated the scalability and efficiency of our simulation environment, which can run on a single low-end GPU device.

![](images/pipelineV2-1.png)

## Abstract
Robotic surgery is a rapidly developing field that can greatly benefit from the automation of surgical tasks. However, training techniques such as Reinforcement Learning (RL) require a high number of task repetitions, which are especially unsafe and impractical to perform on real surgical systems. This
stresses the need for simulated surgical environments, which are not only realistic, but also computationally efficient and scalable. We introduce FF-SRL, a high-performance learning environment for robotic surgery. In FF-SRL both physics simulation and RL policy training reside entirely on a single
GPU. This avoids typical bottlenecks associated with data transfer between the CPU and GPU, leading to an accelerated learning rates. Our results show that FF-SRL reduces the training time of a complex tissue manipulation task by an order of magnitude, down to a couple of minutes, compared to a conventional CPU/GPU simulator. Such speed-up may facilitate the experimentation with RL techniques and contribute to the development of new generation of surgical systems. Hence, we make our code publicly available to the community.

## Features
* Soft-body simulation engine creted with XPBD solver,
* Simulation code written in [nVidia warp](https://github.com/NVIDIA/warp) allowing for fast execution on GPUs and interoperability with e.g. PyTorch for Reinforcement Learning,
* The simulation is written in data-oriented way (data for multiple environments are kept in the same arrays),
* Rendering with simple yet fast ray casting on GPU,
* Bounding Volume Hierarchy creation,
* Scene creation with nVidia OmniVerse (tools for that will be added in feature)
* Instrument grasping implemented and simple topological modifications to be added (cutting, heating),

## Installation
```
git clone https://github.com/SanoScience/FF-SRL.git
conda create -n FF_SRL python=3.9
conda activate FF_SRL
cd FF-SRL
pip install -e .
```

## Examples
![liverTextureCool](https://github.com/SanoScience/FF-SRL/assets/4333336/cbddf759-0223-45ff-8f9d-31d2a1738b42)


An example of liver simulation with a laparoscope can be accessed here:
```
cd FF-SRL/FF_SRL/FF_SRL/tests/
python testLiverRetractionTexture.py
```
In order to move the laparoscope use:
* j to move left
* l to move right
* i to move away (z axis)
* k to move closer (z axis)
* u to move up
* o to move down
* f to clamp the instrument effector
* r to reset the environment

## License
FF-SRL is under ![Apache 2.0](https://github.com/SanoScience/FF-SRL/blob/main/LICENSE).

## Citing
If you use the project in your work, please consider citing it with:
```bibtex
@article{IROS2024,
  author  = {Diego Dall’Alba, Michal Naskret, Sabina Kaminska, Przemyslaw Korzeniowski},
  title   = {FF-SRL: High Performance GPU-Based Simulation For Surgical Robot Learning},
  conference = {IEEE Intelligent Robots and Systems, IROS 2024},
  year    = {2024},
}
