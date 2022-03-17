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
from cupyx.time import repeat
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

from numba import (cuda,
                   uint32,
                   int32, 
                   float32,
                   types, 
                   jit,
                   prange)

from ._kernels import (_block_get_nearest_brute, 
                      _global_get_nearest_brute)

from math import sin, cos, sqrt, asin, sqrt
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

def loop_haversine(lat1, lon1, lat2, lon2):
    
    first_sin = sin((lat2 - lat1) / 2.)
    second_sin = sin((lon2 - lon1) / 2.)
    
    a = first_sin * first_sin + \
        cos(lat1) * \
        cos(lat2) * \
        second_sin * second_sin
    
    a = sqrt(a)
    
    if a > 1.:
        a = 1.
    elif a < 0:
        a = 0.
        
    a = asin(a)
    
    return 2.0 * a

def loop_solve(a, b):
    
    out_idx = np.empty(
        (a.shape[0]), dtype=np.uint32)
    
    out_dist = np.empty(
        (a.shape[0]), dtype=np.float32)
    
    for obs_idx in range(a.shape[0]):
        
        glob_min_dist = 1e11
        glob_min_idx = 0
        
        for ref_idx in range(b.shape[0]):
            
            temp_dist = loop_haversine(
                a[obs_idx, 0],
                a[obs_idx, 1],
                b[ref_idx, 0],
                b[ref_idx, 1])
            
            if temp_dist < glob_min_dist:
                glob_min_dist = temp_dist
                glob_min_idx = ref_idx
        
        out_dist[obs_idx] = glob_min_dist
        out_idx[obs_idx] = glob_min_idx
        
    return out_idx, out_dist

def numpy_haversine(lat1, lon1, lat2, lon2):
    
    return 2.0 * np.arcsin(
        np.sqrt(np.sin((lat2 - lat1) / 2.0)**2 + \
                np.cos(lat1) * \
                np.cos(lat2) * \
                np.sin((lon2 - lon1) / 2.0)**2)
    )
                
def numpy_solve(a, b):
    
    a_broad = a[:,np.newaxis]
    
    temp = numpy_haversine(
        a_broad[:,:,0],
        a_broad[:,:,1],
        b[:,0],
        b[:,1]
    )
    
    np.abs(temp, out=temp)
    out_idx = temp.argmin(axis=1)
    out_dist = temp[np.arange(a.shape[0]), out_idx]    
    
    return out_idx, out_dist

def sklearn_knn_solve(a, b):
    
    knn = NearestNeighbors(
        algorithm="brute",
        metric="haversine")
    
    knn.fit(b)
    
    out_dist_sklrn, out_idx_sklrn = \
    knn.kneighbors(
        a, 
        n_neighbors=1, 
        return_distance=True)
    
    return (out_idx_sklrn.reshape(a.shape[0]),
            out_dist_sklrn.reshape(a.shape[0]))

def _numba_cpu_haversine(lat1, lon1, lat2, lon2):

    return 2.0 * asin(
        sqrt(sin((lat2 - lat1) / 2.0)**2 + \
             cos(lat1) * \
             cos(lat2) * \
             sin((lon2 - lon1) / 2.0)**2)
    )

numba_cpu_haversine = jit(       
    nopython=True, fastmath=True)(_numba_cpu_haversine)

@jit(nopython=True)
def numba_cpu_solve(a, b):
    
    out_idx = np.empty(
        (a.shape[0]), dtype=np.uint32)
    
    out_dist = np.empty(
        (a.shape[0]), dtype=np.float32)
    
    for obs_idx in range(a.shape[0]):
        
        glob_min_dist = 1e11
        glob_min_idx = 0
        
        for ref_idx in range(b.shape[0]):
            
            temp_dist = numba_cpu_haversine(
                a[obs_idx,0],
                a[obs_idx, 1],
                b[ref_idx, 0],
                b[ref_idx, 1])
            
            if temp_dist < glob_min_dist:
                glob_min_dist = temp_dist
                glob_min_idx = ref_idx
        
        out_dist[obs_idx] = glob_min_dist
        out_idx[obs_idx] = glob_min_idx
        
    return out_idx, out_dist

@jit(nopython=True, parallel=True)
def numba_multi_cpu_solve(a, b):
    
    out_idx = np.empty(
        (a.shape[0]), dtype=np.uint32)
    
    out_dist = np.empty(
        (a.shape[0]), dtype=np.float32)
    
    for obs_idx in prange(a.shape[0]):
        
        glob_min_dist = 1e11
        glob_min_idx = 0
        
        for ref_idx in range(b.shape[0]):
            
            temp_dist = numba_cpu_haversine(
                a[obs_idx,0],
                a[obs_idx, 1],
                b[ref_idx, 0],
                b[ref_idx, 1])
            
            if temp_dist < glob_min_dist:
                glob_min_dist = temp_dist
                glob_min_idx = ref_idx
        
        out_dist[obs_idx] = glob_min_dist
        out_idx[obs_idx] = glob_min_idx
        
    return out_idx, out_dist

def cupy_haversine(lat1, lon1, lat2, lon2):
    
    return 2.0 * cp.arcsin(
        cp.sqrt(cp.sin((lat2 - lat1) / 2.0)**2 + \
                cp.cos(lat1) * \
                cp.cos(lat2) * \
                cp.sin((lon2 - lon1) / 2.0)**2)
    )

def cupy_solve(a, b):
    
    a_broad = a[:,cp.newaxis]
    
    temp = cupy_haversine(
        a_broad[:,:,0],
        a_broad[:,:,1],
        b[:,0],
        b[:,1]
    )
    
    cp.abs(temp, out=temp, dtype=np.float32)
    out_idx = temp.argmin(axis=1, dtype=np.int32)
    out_dist = temp[cp.arange(a.shape[0]), out_idx]    
    
    return out_idx, out_dist 

def cuml_knn_solve(a, b):
    
    cuknn = cuNearestNeighbors(
        algorithm="brute",
        metric="haversine")
    
    cuknn.fit(b)
    
    out_dist_cuml, out_idx_cuml = \
    cuknn.kneighbors(
        a, 
        n_neighbors=1, 
        return_distance=True)
    
    return (out_idx_cuml.reshape(a.shape[0]),
            out_dist_cuml.reshape(a.shape[0]))

sig_block_get_nearest_brute = \
    "void(float32[:,:], float32[:,:], uint32[:,:], float32[:,:])"

block_min_reduce = cuda.jit(
    sig_block_get_nearest_brute,        
    fastmath=True)(_block_get_nearest_brute)

sig_global_get_nearest_brute = \
    "void(float32[:,:], uint32[:,:], float32[:], uint32[:])"

global_min_reduce = cuda.jit(
    sig_global_get_nearest_brute,        
    fastmath=True)(_global_get_nearest_brute)

def numba_cuda_solve(d_obs, d_ref):
    
    d_out_idx = cuda.device_array(
        (d_obs.shape[0],), 
        dtype=np.uint32)
    
    d_out_dist = cuda.device_array(
        (d_obs.shape[0],), 
        dtype=np.float32)     
    
    d_block_idx = cuda.device_array(
        (d_out_idx.shape[0], 32), 
        dtype=np.uint32)

    d_block_dist = cuda.device_array(
        (d_out_idx.shape[0], 32), 
        dtype=np.float32)           

    bpg = 32, 108
    tpb = 32, 16

    block_min_reduce[bpg, tpb](
        d_ref, 
        d_obs, 
        d_block_idx,
        d_block_dist)   

    bpg = (1, 108*20)
    tpb = (32, 16)        

    global_min_reduce[bpg, tpb](
        d_block_dist, 
        d_block_idx, 
        d_out_dist, 
        d_out_idx)   
        
    cuda.synchronize()
    
    return d_out_idx, d_out_dist
