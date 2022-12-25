---
title: Neural Fields are cool
date: 2022-10-18
---

Ok so this is going to be a fairly generic type of post. A year or so ago, I was really fascinated by the propsects of deep learning on point clouds in fairly arbitrary dimensional euclidean spaces. 
Most of my fascination with such a topic was centered on it's application to deep representation learning for 3D synthesis while minizing computational load by sparsifying into point sets, or euclidean coordinates as opposed to regular voxel grids (which are a lot more "memory-sucking"). The pointnet architecure seemed to me, like a very seminal advance in the space of deep learning on 3D objects.

A major glut-point afflicting a lot of contemporary 3D deep-learning is the insane memory, and downstream computationl costs of rendering 3D objects of any size. Think 30GB for 5mins of high-quality 3D rendering. A lot of current approaches, at least up until 2022 DL compression boom (will speak more about this later) were operating on a fairly inefficient paradigm. 

We all know about good old fields. Generally speaking, in physics - or in general, a field is a way to represent quantities that are defined for every point in space. More specifically, to help ground intuitions, we can refer to it as a vector field. 
Cool examples that are intuitive are the eletromagentic field and light as a field quantity. Or the gravitational field and gravity (still pretty uunclear what do) being a quantity. So, I guess you see how we build this up now to *radiance* fields. 
Now this is where it starts getting fun. 
---

We all love graphics right? good ol graphics. Where virtually every other subdiscipline must gift alms surfeit in servitude. Radiance fields in the domain of graphics are primarily used to map light scattering over a continuos space leveraging electromagnetic field properties.
You could imagine a radiance field as a subset of an EM field. So for optimality sake, we'd like an algorithm that can efficiently map the set of input taken onto a smaller dimensional space of color pixel density distributions.

More precisely:

