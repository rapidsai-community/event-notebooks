# Accelerated Numerical Computing with RAPIDS

This repository contains source code referenced in this GTC Spring 2022 session presented by Matthew Penn at NVIDIA:

Evaluating Your Options for Accelerated Numerical Computing in Pure Python [S41645]

Abstract - We'll tackle a generic n-dimensional numerical computing problem and explore performance and trade-offs between popular frameworks using open-source Jupyter notebook examples. Most data science practitioners perform their work in Python because of its high-level abstraction and rich set of numerical computing libraries. But the choice of library and methodology is driven by complexity-impacting constraints like problem size, latency, memory, SWAP, hardware, and others. To that end, we show that a wide selection of GPU-accelerated libraries ([RAPIDS](https://rapids.ai/start.html), [CuPy](https://cupy.dev/), [Numba](https://numba.pydata.org/numba-doc/latest/cuda/index.html), [Dask](https://docs.rapids.ai/api/dask-cuda/stable/api.html), [cuNumeric](https://github.com/nv-legate/cunumeric)), including the development of hand-tuned CUDA kernels, are accessible to data scientists without ever leaving Python.

### **Use Case Overview**

In these notebook examples, we survey techniques for numerical computing in pure python. We leverage a proxy geospatial nearest neighbor problem to guide us through an evaluation of several libraries and methodologies. In this use case, aim to resolve geospatial observations to their nearest reference points with an added complexity. Our complication adds dynamics to the problem allowing each reference point to move and the set of observations to change on a reoccurring basis. These complexities imply a need to recompute each nearest neighbor at each timestep -- emphasizing the need for high performance techiques. 

*Note - Benchmarks were generated on a DGX Station A100 320GB*

### **Contents**

- examples/single-cpu-gpu.ipynb - single threaded CPU and single GPU techniques on moderate sample problem
- examples/multi-cpu-numba.ipynb - parallel CPU technique
- examples/multi-gpu-threading-rmm-numba.ipynb - multi-GPU technique using Threading + RMM + Numba CUDA
- examples/multi-gpu-dask-cudf-numba.ipynb - multi-GPU technique using Dask cuDF + Numba CUDA
- examples/visualization.ipynb - sample visualization of the dynamic geospatial problem
- examples/src - source code for imported functions

### **Build Container**<br>

```sh
sudo docker build . -t numerical-computing:gtc-examples
```

### **Run Container**<br>

```sh
sudo docker run -it --rm --runtime=nvidia \
    -v ${PWD}/:/numerical-computing \
    -p 8888:8888 \
    -p 8787:8787 \
    -p 8686:8686 \
    -w /numerical-computing numerical-computing:gtc-examples
```

Once running, the container will expose JupyterLab at port 8888.

