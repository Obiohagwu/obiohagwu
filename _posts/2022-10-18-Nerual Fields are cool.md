---
title: Neural Fields are cool
date: 2022-10-18
---

Ok so this is going to be a fairly generic type of post. A year or so ago, I was really fascinated by the propsects of deep learning on point clouds in fairly arbitrary dimensional euclidean spaces. 
Most of my fascination with such a topic was centered on it's application to deep representation learning for 3D synthesis while minizing computational load by sparsifying into point sets, or euclidean coordinates as opposed to regular voxel grids (which are a lot more "memory-sucking"). The pointnet architecure seemed to me, like a very seminal advance in the space of deep learning on 3D objects.

A major glut-point afflicting a lot of contemporary 3D deeplearning is the insane memory, and downstream computationl costs of rendering 3D objects of any size. Think 30GB for 5mins of high-quality 3D rendering. A lot of current approaches, at least up until 2022 DL compression boom (will speak more about this later)