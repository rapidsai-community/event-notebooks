"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import cupy as cp
from src.simulator import generate_geos, permutate
from src.solvers import numba_cuda_solve
from src.utils import zeropad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def multi_plot(n_obs, n_ref, n_steps=3, 
               x_vel=100000, y_vel=100000, 
               timestep=10, random_state=1,
               h=0.02, cmap="Dark2"):
    
    lat_min, lat_max = -cp.pi/2, cp.pi/2
    lon_min, lon_max = -cp.pi, cp.pi    
    
    d_lats, d_lons = cp.meshgrid(
        cp.arange(lat_min, lat_max, h), 
        cp.arange(lon_min, lon_max, h)) 
    
    h_lats = d_lats.get()
    h_lons = d_lons.get()
    
    d_grid = \
        cp.c_[d_lats.ravel(), 
              d_lons.ravel()].astype(np.float32)    
    
    f, ax = plt.subplots(
        1, n_steps, sharey=True, figsize=(25,5))
    
    for step in range(n_steps):
        
        d_obs = generate_geos(
            n_obs, random_state=step
        )
        
        if step == 0:
            d_ref = generate_geos(
                n_ref, random_state=step+n_steps
            )
            
        else:
            d_ref = permutate(
                d_ref, 
                x_vel=x_vel, 
                y_vel=y_vel, 
                timestep=timestep, 
                random_state=step+n_steps
            )
                
        d_obs_Z, _ = numba_cuda_solve(
            d_obs,
            d_ref
        )
        
        d_grid_Z = numba_cuda_solve(
            d_grid, 
            d_ref
        )[0].reshape(d_lats.shape)
        
        h_obs = d_obs.get()
        h_obs_Z = d_obs_Z.copy_to_host()
        h_grid_Z = d_grid_Z.copy_to_host()
        
        ax[step].contourf(
            h_lons, 
            h_lats, 
            h_grid_Z,
            cmap="Dark2",
        )
        
        sns.scatterplot(
            ax=ax[step],
            x=h_obs[:, 1],
            y=h_obs[:, 0],  
            hue=h_obs_Z,
            cmap="Dark2",
            alpha=1.0,
            edgecolor="black",
            s=1
        )

        ax[step].get_legend().remove()       
        ax[step].set_xlim(lon_min, lon_max)
        ax[step].set_ylim(lat_min, lat_max)        
        
        ax[step].set_title(
            "Nearest Neighbor Decision Boundaries: Step {}".format(step)
        )

        ax[step].set_xlabel("Longitude (Radians)")
        ax[step].set_ylabel("Latitude (Radians)")        
    
    plt.show()