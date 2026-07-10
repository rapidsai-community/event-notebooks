# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np

def generate_geos(n_samples=1000, 
                  random_state=1):
    
    cp.random.seed(random_state)
    
    lat_ext = (-90., 90.)
    lon_ext = (-180., 180.)   
    
    R = 6373.
    
    lats = cp.random.uniform(
        low=lat_ext[0], 
        high=lat_ext[1], 
        size=n_samples, 
        dtype=np.float32
    )
    
    lons = cp.random.uniform(
        low=lon_ext[0], 
        high=lon_ext[1], 
        size=n_samples, 
        dtype=np.float32)
    
    geopoints = cp.empty(
        shape=(n_samples, 6), 
        dtype=np.float32
    )
    
    geopoints[:,0] = lats
    geopoints[:,1] = lons
    geopoints[:,2] = lats * cp.pi / 180
    geopoints[:,3] = lons * cp.pi / 180
    geopoints[:,4] = (lats * cp.pi / 180) * R
    geopoints[:,5] = (lons * cp.pi / 180) * R
    
    return geopoints[:,2:4]

def lat_bounce(rad_x0, rad_xs):
    
    rad_temp = rad_x0 + rad_xs
    
    pi_o_2 = cp.pi/2
    
    bounce_d_pts = rad_temp > pi_o_2
    
    temp_d = rad_temp[bounce_d_pts] > pi_o_2
    rad_temp[bounce_d_pts] = temp_d - pi_o_2
    
    bounce_u_pts = rad_temp < -pi_o_2
    
    temp_u = rad_temp[bounce_u_pts] < -pi_o_2
    rad_temp[bounce_u_pts] = pi_o_2 - temp_u
    
    return rad_temp

def lon_bounce(rad_y0, rad_ys):
    
    rad_temp = rad_y0 + rad_ys
    
    bounce_l_pts = rad_temp > cp.pi
    
    temp_l = rad_temp[bounce_l_pts] - cp.pi
    rad_temp[bounce_l_pts] = temp_l - cp.pi
    
    bounce_r_pts = rad_temp < -cp.pi
    temp_r = rad_temp[bounce_r_pts] + cp.pi
    rad_temp[bounce_r_pts] = cp.pi - temp_r
    
    return rad_temp

def permutate(geopoints, x_vel=100, y_vel=100, timestep=100, random_state=1):
    
    cp.random.seed(random_state)
    
    perm_geopoints = cp.empty(geopoints.shape, dtype=np.float32)
    
    rad_xs_0 = geopoints[:,1]
    rad_ys_0 = geopoints[:,0]
    
    radius = 6378100 # meters
    
    v_xs = cp.random.uniform(
        low=-x_vel, high=x_vel, size=geopoints.shape[0]
    )
    
    v_ys = cp.random.uniform(
        low=-y_vel, high=y_vel, size=geopoints.shape[0]
    )       
        
    d_xs = v_xs * timestep
    d_ys = v_ys * timestep
    
    rad_xs = d_xs / radius
    rad_ys = d_ys / radius
    
    rad_xs_1 = lon_bounce(rad_xs_0, rad_xs)
    rad_ys_1 = lat_bounce(rad_ys_0, rad_ys)
    
    perm_geopoints[:,0] = rad_ys_1
    perm_geopoints[:,1] = rad_xs_1
    
    return perm_geopoints