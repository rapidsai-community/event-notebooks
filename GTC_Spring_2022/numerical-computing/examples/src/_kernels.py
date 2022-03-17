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

from numba import (cuda, 
                   int32, 
                   uint32, 
                   float32, 
                   types)

from math import sin, cos, sqrt, asin

@cuda.jit(device=True, inline=True)
def _warp_min_reduce_idx_unrolled(val, idx):
    
    mask  = 0xffffffff    
        
    shfl_val = cuda.shfl_down_sync(
        mask, val, 16)
    
    shfl_idx = cuda.shfl_down_sync(
        mask, idx, 16)

    if val > shfl_val:
        val = shfl_val
        idx = shfl_idx
        
    shfl_val = cuda.shfl_down_sync(
        mask, val, 8)
    
    shfl_idx = cuda.shfl_down_sync(
        mask, idx, 8)

    if val > shfl_val:
        val = shfl_val
        idx = shfl_idx        
        
    shfl_val = cuda.shfl_down_sync(
        mask, val, 4)
    
    shfl_idx = cuda.shfl_down_sync(
        mask, idx, 4)

    if val > shfl_val:
        val = shfl_val
        idx = shfl_idx         
        
    shfl_val = cuda.shfl_down_sync(
        mask, val, 2)
    
    shfl_idx = cuda.shfl_down_sync(
        mask, idx, 2)

    if val > shfl_val:
        val = shfl_val
        idx = shfl_idx         
        
    shfl_val = cuda.shfl_down_sync(
        mask, val, 1)
    
    shfl_idx = cuda.shfl_down_sync(
        mask, idx, 1)

    if val > shfl_val:
        val = shfl_val
        idx = shfl_idx

    return val, idx

def _block_get_nearest_brute(
    coord1, coord2, block_idx, block_dist):
    
    """
    GPU accelerated pairwise distance comparisons in single
    precision.
    """    
    
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    seed = float32(1e11)
        
    for i in range(starty, coord2.shape[0], stridey):    
        
        b_min_val = seed
        b_min_idx = uint32(0)
        coord2_i_0 = coord2[i,0]
        
        for j in range(startx, coord1.shape[0], stridex):

            coord1_j_0 = coord1[j,0]
        
            first_sin = sin(
                (coord2_i_0 - coord1_j_0) * float32(0.5))
            
            second_sin = sin(
                (coord2[i,1] - coord1[j, 1]) * float32(0.5))            

            local_val = float32(2.0) * asin(
                sqrt(
                    first_sin * first_sin + \
                    cos(coord1_j_0) * \
                    cos(coord2_i_0) * \
                    second_sin * second_sin)
            )            
            
            if local_val < b_min_val:
                b_min_val = local_val
                b_min_idx = j
                
                
        b_min_val, b_min_idx = \
            _warp_min_reduce_idx_unrolled(
            b_min_val, b_min_idx)

        if cuda.laneid == 0:
            block_dist[i, cuda.blockIdx.x] = b_min_val
            block_idx[i, cuda.blockIdx.x] = b_min_idx
            
def _global_get_nearest_brute(
    block_dist, block_idx, out_dist, out_idx):        
        
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    seed = float32(1e11)
    
    for i in range(starty, out_dist.shape[0], stridey):
        
        g_min_dist = seed
        g_min_idx = 0
        
        for j in range(startx, block_idx.shape[1], stridex):
            
            local_dist = block_dist[i, cuda.threadIdx.x]
            
            if local_dist < g_min_dist:
                g_min_dist = local_dist
                g_min_idx = block_idx[i, cuda.threadIdx.x]
        
        g_min_dist, g_min_idx = \
            _warp_min_reduce_idx_unrolled(
            g_min_dist, g_min_idx)
        
        if cuda.laneid == 0:
            out_dist[i] = g_min_dist
            out_idx[i] = g_min_idx
