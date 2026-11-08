/*********************************************************************************
 * Copyright (c) 2018, Chong Peng. All rights reserved.

 * <LOQUAT> Three-dimensional GPU-accelerated SPH solver for geotechnical modeling

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    1. Redistributions of source code must retain the copyright
 *       notice, this list of conditions and the following disclaimer.
 *    2. The origin of this software must not be misrepresented; you must
 *       not claim that you wrote the original software.
 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
 * EVENT SHALL THE HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE.
 *********************************************************************************/

#include <helper_cuda.h>
#include "loquat.h"
#include "constitutive_model.h"

static __constant__ Parameters par;

static __inline__ __device__ bool cuCheckOutDomainPeriod(int4& celli, scalar4& pospres_i) {
    scalar3 pos_i = make_scalar3(pospres_i);
    scalar3 pos_abs_i = pos_i + make_int3(celli) * (2.0 * par.h); //Transition from relative position to absolute position
    bool period_left = Dist2Plane(pos_abs_i, par.planeleft) < 0.0;
    bool period_right = Dist2Plane(pos_abs_i, par.planeright) < 0.0;
    if (period_left || period_right) {
        scalar shift_length = abs(par.planeleft.w + par.planeright.w);
        scalar3 shift_vector = period_left ? make_scalar3(par.planeleft) * shift_length : make_scalar3(par.planeright) * shift_length;
        pos_abs_i = pos_abs_i + shift_vector;
        int3 cell_pb_i = make_int3(pos_abs_i * (0.5 * par.i_h));
        pos_i = pos_abs_i - cell_pb_i * (2.0 * par.h);
        celli = make_int4(cell_pb_i, celli.w);
        pospres_i = make_scalar4(pos_i, pospres_i.w);
    }
}

static __inline__ __device__ bool cuCheckOutDomain(const int3 celli) {
    if ((!(celli >= make_int3(0))) || (!(celli < par.cell_num))) return true;
    else return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuPredictorMaterial(int4* codes, scalar4* pospres, scalar4* velrhop, scalar4* idmassidx, tensor2d* str, tensor2d* stn,
    scalar4* accad, scalar4* spin, tensor2d* acestn){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = par.npb + index;
    if(index < par.npm && codes[i].y == NonFix) {
        // Update the velocity and position
        velrhop[i] = velrhop[i] + accad[i] * (0.5f * par.dt);
        pospres[i] = pospres[i] + make_scalar3(velrhop[i]) * (0.5f * par.dt);
        velrhop[i].w = max(velrhop[i].w, par.model_par[codes[i].w][MATERIAL_SIZE_MAX - 2]);
        velrhop[i].w = min(velrhop[i].w, par.model_par[codes[i].w][MATERIAL_SIZE_MAX-1]);

        // Update the stress state
        tensor2d dstn = acestn[i] * (0.5f * par.dt);
        stn[i] = stn[i] + dstn;
        tensor2d sigma = str[i];
        scalar4 omega = spin[i];   

        bool yld = 0;//yield in DP or not
        scalar M_crt = 0;//critical state parameter

        if (codes[i].z == DP_EPP)cuMatModDP(0.5f * par.dt, omega, acestn[i], sigma, par.model_par[codes[i].w], yld);
        else if (codes[i].z == MuI_DP_EPP) cuMatModMuIDP(0.5f * par.dt, omega, acestn[i], sigma, par.model_par[codes[i].w]);
        else if(codes[i].z == HypoBasic) cuMatModHypo(0.5f * par.dt, omega, acestn[i], sigma, par.model_par[codes[i].w]);
        else if (codes[i].z == HypoCS) cuMatModHypo(0.5f * par.dt, velrhop[i].w, omega, acestn[i], sigma, par.model_par[codes[i].w], M_crt);
        str[i] = sigma;

        if (yld) {
            //idmassidx[i].x = 101;//if yield id 101
        }
        else {
            idmassidx[i].x = 100;
        }
    }
}

void LoquatSph::Predictor(){
    checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

    // Save the information about the velocity and position at the beginning of the calculation step
    checkCudaErrors(cudaMemcpyAsync(pospres_ini, pospres, par_host->np * sizeof(scalar4), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyAsync(velrhop_ini, velrhop, par_host->np * sizeof(scalar4), cudaMemcpyDeviceToDevice));
    if(par_host->npm != 0){
        checkCudaErrors(cudaMemcpyAsync(str_ini, str, (par_host->npb + par_host->npm) * sizeof(tensor2d), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyAsync(stn_ini, stn, (par_host->npb + par_host->npm) * sizeof(tensor2d), cudaMemcpyDeviceToDevice));
    }
    dim3 block(128, 1);
    if(par_host->npm != 0){
        dim3 cuda_grid_size = dim3((par_host->npm + block.x - 1) / block.x, 1);
        cuPredictorMaterial <<<cuda_grid_size, block, 0>>> (codes, pospres, velrhop, idmassidx, str, stn, accad, spin, acestn);
        getLastCudaError("cuPredictorMaterial failed...     \n");
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void cuCorrectorBound(int4* codes, int4* cellpos, scalar4* pospres, scalar4* normal, scalar4* velrhop, scalar4* idmassidx, tensor2d* str, tensor2d* stn,
    scalar4* pospres_ini, scalar4* velrhop_ini, tensor2d* str_ini, tensor2d* stn_ini, scalar4* accad, scalar4* spin, tensor2d* acestn) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index;
    if (index < par.npb && codes[i].y == NonFix) {
        // Velocity and position update
        scalar3 vel = par.mv[codes[i].w].trans_vel;
        if (par.mv[codes[i].w].is_rotat&& par.mv[codes[i].w].pattern!=Circle) {
            scalar3 pos_i = make_scalar3(pospres[i]);
            int3 celli = make_int3(cellpos[i]);
            scalar3 pos_abs_i = pos_i + celli * (2.0 * par.h);
            scalar3 pos_abs_diff_i_rp = pos_abs_i - par.mv[codes[i].w].pos_rp;
            vel = vel + CrossProduct(par.mv[codes[i].w].rotat_vel, pos_abs_diff_i_rp);  
            tensor2dful rotat_matrix = GetRodriguesMatrix(par.mv[codes[i].w].rotat_vel * par.dt);
            scalar3 normal_i = make_scalar3(normal[i]);
            normal_i = DotProduct(rotat_matrix, normal_i);
            normal[i] = make_scalar4(normal_i, normal[i].w);
        }
        pospres[i] = pospres_ini[i] + vel * par.dt;

        int3 dcell = make_int3(0);
        dcell.x = int(pospres[i].x >= 2.0 * par.h) - int(pospres[i].x < 0.0);
        dcell.y = int(pospres[i].y >= 2.0 * par.h) - int(pospres[i].y < 0.0);
        dcell.z = int(pospres[i].z >= 2.0 * par.h) - int(pospres[i].z < 0.0);

        pospres[i] = pospres[i] - dcell * (2.0 * par.h);
        cellpos[i] = cellpos[i] + dcell;
    }
}

__global__ void cuCorrectorMaterial(int4* codes, int4* cellpos, scalar4* pospres, scalar4* velrhop, scalar4* idmassidx, tensor2d* str, tensor2d* stn,
scalar4* pospres_ini, scalar4* pospres_ori, scalar4* velrhop_ini, tensor2d* str_ini, tensor2d* stn_ini, scalar4* accad, scalar4* spin, tensor2d* acestn){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = par.npb + index;
    if (index < par.npm && codes[i].y == NonFix) {
        // Velocity and position update
        velrhop[i] = velrhop_ini[i] + accad[i] * par.dt;
        pospres[i] = pospres_ini[i] + make_scalar3(velrhop[i] + velrhop_ini[i]) * (0.5f * par.dt);
        velrhop[i].w = max(velrhop[i].w, par.model_par[codes[i].w][MATERIAL_SIZE_MAX - 2]);
        velrhop[i].w = min(velrhop[i].w, par.model_par[codes[i].w][MATERIAL_SIZE_MAX - 1]);

        int3 dcell = make_int3(0);
        dcell.x = int(pospres[i].x >= 2.0 * par.h) - int(pospres[i].x < 0.0);
        dcell.y = int(pospres[i].y >= 2.0 * par.h) - int(pospres[i].y < 0.0);
        dcell.z = int(pospres[i].z >= 2.0 * par.h) - int(pospres[i].z < 0.0);

        pospres[i] = pospres[i] - dcell * (2.0 * par.h);
        cellpos[i]= cellpos[i] + dcell;

        if (par.periodbound)cuCheckOutDomainPeriod(cellpos[i], pospres[i]);

        // Stress state update
        tensor2d sigma = str_ini[i];
        scalar4 omega = spin[i];
        tensor2d dstn = acestn[i] * par.dt;
        stn[i] = stn_ini[i] + dstn;

        bool yld = 0;//yield in DP or not
        scalar M_crt = 0;//critical state parameter

        if(codes[i].z == DP_EPP) cuMatModDP(par.dt, omega, acestn[i], sigma, par.model_par[codes[i].w], yld);
        else if (codes[i].z == MuI_DP_EPP) cuMatModMuIDP(par.dt, omega, acestn[i], sigma, par.model_par[codes[i].w]);
        else if (codes[i].z == HypoBasic) cuMatModHypo(par.dt, omega, acestn[i], sigma, par.model_par[codes[i].w]);
        else if (codes[i].z == HypoCS) cuMatModHypo(par.dt, velrhop[i].w, omega, acestn[i], sigma, par.model_par[codes[i].w], M_crt);
        str[i] = sigma;

        //If yield id 101
        if (yld) {
            //idmassidx[i].x = 101;
        }
        else {
            idmassidx[i].x = 100;
        }

        //Output critical state parameter in idx
        idmassidx[i].w = M_crt;

        // Check if particle i moving out of calculation domain
        if (cuCheckOutDomain(make_int3(cellpos[i]))) {
            cellpos[i].w = par.ncell_delete_m;
            idmassidx[i].x = 999.0;
        }

        //Record accumulated displacement in accad
        if (par.time <= par.dt) {
            pospres_ori[i] = make_scalar4(0);
        }
        pospres_ori[i] = pospres_ori[i] + make_scalar3(velrhop[i]) * (par.dt);
        accad[i] = pospres_ori[i];
    }
}

void LoquatSph::Corrector(){
    checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));
    dim3 block(128, 1);
    if (par_host->n_motion > 0) {
        dim3 cuda_grid_size = dim3((par_host->npb + block.x - 1) / block.x, 1);
        cuCorrectorBound << <cuda_grid_size, block, 0, 0 >> > (codes, cellpos, pospres, normal_m, velrhop, idmassidx, str, stn, pospres_ini, velrhop_ini, str_ini, stn_ini, accad, spin, acestn);
        getLastCudaError("cuCorrectorBound failed...     \n");
    }
    if(par_host->npm != 0){
        dim3 cuda_grid_size = dim3((par_host->npm + block.x - 1) / block.x, 1);
        cuCorrectorMaterial <<<cuda_grid_size, block, 0, 0>>> (codes, cellpos, pospres, velrhop, idmassidx, str, stn, pospres_ini, pospres_ori, velrhop_ini, str_ini, stn_ini, accad, spin, acestn);
        getLastCudaError("cuCorrectorMaterial failed...     \n");
    }
}