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
from pynvml.smi import nvidia_smi
from .simulator import generate_geos
from .solvers import loop_solve, loop_haversine
import numpy as np

def query_available_memory():   
    
    nvsmi = nvidia_smi.getInstance()
    available_mem = nvsmi.DeviceQuery(
        'memory.free, memory.total, index')

    for i, dev in enumerate(available_mem['gpu']):
        
        if i == 0:
            free_mem = dev['fb_memory_usage']["free"]
            device = int(dev['minor_number'])
        
        if dev['fb_memory_usage']["free"] > free_mem:
            free_mem = dev['fb_memory_usage']["free"]
            device = int(dev['minor_number'])
                                
    free_mem = int((free_mem * 1e6) / 256) * 256
    
    return (free_mem, device)

def check_accuracy(d_obs, d_ref, out_idx, out_dist, atol=1e-6):  
    
    if isinstance(d_obs, cp.ndarray):
        h_obs = d_obs.get()
    else:
        h_obs = d_obs
        
    if isinstance(d_ref, cp.ndarray):
        h_ref = d_ref.get()        
    else:
        h_ref = d_ref
    
    # baseline/ground truth
    out_idx_loop, out_dist_loop = loop_solve(h_obs, h_ref)
    
    out_idx = cp.asnumpy(out_idx)  
    out_dist = cp.asnumpy(out_dist) 
        
    # validate dists
    valid = np.allclose(out_dist_loop, out_dist, atol=atol)
    
    if not valid:
        return valid
    
    # validate idxs
    valid = np.allclose(out_idx_loop, out_idx)
    
    if valid:
        return True
    
    else:
        # check for rare exact solutions with different indices
        missed_idxs = np.where(out_idx_loop != out_idx)[0]
                                
        dists_match = True
        
        for idx in missed_idxs:
            
            obs_point = h_obs[idx]
            
            out_idx_loop_match = h_ref[out_idx_loop[idx]]
            out_idx_match = h_ref[out_idx[idx]]
            
            dist0 = loop_haversine(
                obs_point[0],
                obs_point[1],
                out_idx_loop_match[0],
                out_idx_loopmatch[1])   
            
            dist1 = loop_haversine(
                obs_point[0],
                obs_point[1],
                out_idx_match[0],
                out_idx_match[1]) 
                                                
            if abs(dist0 - dist1) > atol:
                dists_match = False
                
        return dists_match  
    
def check_accuracy_h2h(
    a, b, out_idx0, out_dist0, out_idx1, out_dist1, atol=1e-6):
        
    a = cp.asnumpy(a)
    b = cp.asnumpy(b)
        
    out_idx0 = cp.asnumpy(out_idx0)  
    out_dist0 = cp.asnumpy(out_dist0)
    
    out_idx1 = cp.asnumpy(out_idx1)  
    out_dist1 = cp.asnumpy(out_dist1) 
        
    # validate dists
    valid = np.allclose(out_dist0, out_dist1, atol=atol)
    
    print("stage 1", valid)
    
    if not valid:
        return valid
    
    # validate idxs
    valid = np.allclose(out_idx0, out_idx1)
    print("stage 2", valid)
    
    if valid:
        return True
    
    else:
        # check for rare exact solutions with different indices
        missed_idxs = np.where(out_idx0 != out_idx1)[0]
                                
        dists_match = True
        
        for idx in missed_idxs:
            
            a_point = a[idx]
            
            out_idx0_match = b[out_idx0[idx]]
            out_idx1_match = b[out_idx1[idx]]
            
            dist0 = loop_haversine(
                a_point[0],
                a_point[1],
                out_idx0_match[0],
                out_idx0_match[1])   
            
            dist1 = loop_haversine(
                a_point[0],
                a_point[1],
                out_idx1_match[0],
                out_idx1_match[1]) 
                                                
            if abs(dist0 - dist1) > atol:
                dists_match = False
                
        return dists_match    
    
def zeropad(value, n_steps):
    return str(value).zfill(len(str(n_steps)))    