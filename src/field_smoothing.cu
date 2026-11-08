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

static __constant__ Parameters par;

__inline__ __device__ void cuStressRegularizationShepardLoopSolution(const int part_beg, const int part_end, const int3 cell_offset, const scalar3 pos_i,
	scalar4* pospres, scalar4* velrhop, scalar4* idmassidx, tensor2d* str, tensor2d& str_shepard, scalar& sumker){
	for(int j = part_beg; j < part_end; j++){
		scalar3 pos_j = make_scalar3(pospres[j]);
		scalar3 dr = pos_i - pos_j - cell_offset * (2.0 * par.h);;
		scalar r = norm3df(dr.x, dr.y, dr.z);
		scalar q = r * par.i_h;
		if(q < 2.0f){
			scalar ker = GetKernel(par.kernel_normalization_par, q);
			scalar vol = idmassidx[j].y / velrhop[j].w;
			sumker += ker * vol;
			str_shepard = str_shepard + str[j] * (ker * vol);
		}
	}
}

__inline__ __device__ void cuStressRegularizationShepardLoop(const int3 celli, const scalar3 pos_i, scalar4* pospres, scalar4* velrhop,
	scalar4* idmassidx, tensor2d* str, const int* cell_beg, const int* cell_end, tensor2d& str_shepard, scalar& sumker) {
	for (int cz = -1; cz <= 1; cz++) {
		for (int cy = -1; cy <= 1; cy++) {
			for (int cx = -1; cx <= 1; cx++) {
				int3 cell_offset = make_int3(cx, cy, cz);
				int3 cellj = make_int3(celli.x + cx, celli.y + cy, celli.z + cz);
				if ((!(cellj >= make_int3(0))) || (!(cellj < par.cell_num)))continue;

				int cell_idx = cellj.x * par.cell_num.y * par.cell_num.z + cellj.y * par.cell_num.z + cellj.z;
				cell_idx += par.ncell_material;
				cuStressRegularizationShepardLoopSolution(cell_beg[cell_idx], cell_end[cell_idx], cell_offset, pos_i, pospres, velrhop, idmassidx, str, str_shepard, sumker);
			}
		}
	}
}

__inline__ __device__ void cuStressRegularizationShepardLoopPeriod(const int3 celli, const scalar3 pos_i, scalar4* pospres, scalar4* velrhop,
	scalar4* idmassidx, tensor2d* str, const int* cell_beg, const int* cell_end, tensor2d& str_shepard, scalar& sumker) {
	scalar3 pos_abs_i = pos_i + celli * (2.0 * par.h); //Transition from relative position to absolute position
	bool period_left = AbsDist2Plane(pos_abs_i, par.planeleft) < 2.0 * par.h;
	bool period_right = AbsDist2Plane(pos_abs_i, par.planeright) < 2.0 * par.h;
	if (period_left || period_right) {
		scalar shift_length = abs(par.planeleft.w + par.planeright.w);
		scalar3 shift_vector = period_left ? make_scalar3(par.planeleft) * shift_length : make_scalar3(par.planeright) * shift_length;
		scalar3 pos_abs_pb_i = pos_abs_i + shift_vector;
		int3 cell_pb_i = make_int3(pos_abs_pb_i * (0.5 * par.i_h));
		scalar3 pos_pb_i = pos_abs_pb_i - cell_pb_i * (2.0 * par.h);
		cuStressRegularizationShepardLoop(cell_pb_i, pos_pb_i, pospres, velrhop, idmassidx, str, cell_beg, cell_end, str_shepard, sumker);
	}
}

__global__ void cuStressRegularizationShepard(int4* cellpos, scalar4* pospres, scalar4* velrhop, scalar4* idmassidx, tensor2d* str,
	tensor2d* tensor2d_cache, const int* cell_beg, const int* cell_end){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < par.npm)	{
		int i = index + par.npb;
		scalar3 pos_i = make_scalar3(pospres[i]);
		int3 celli = make_int3(cellpos[i]);		
		tensor2d str_shepard = make_tensor2d(0.0);
		scalar sumker = 0.0f;

		cuStressRegularizationShepardLoop(celli, pos_i, pospres, velrhop, idmassidx, str, cell_beg, cell_end, str_shepard, sumker);
		if(par.periodbound)cuStressRegularizationShepardLoopPeriod(celli, pos_i, pospres, velrhop, idmassidx, str, cell_beg, cell_end, str_shepard, sumker);
		
		if(sumker != 0.f) str_shepard = str_shepard / sumker;
		tensor2d_cache[i] = str_shepard;
	}
}

void LoquatSph::StressRegularization(){
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));
	dim3 block(128, 1);
	if(par_host->npm != 0){
		dim3 cuda_grid_size = dim3((par_host->npm + block.x - 1) / block.x, 1);
		cuStressRegularizationShepard <<<cuda_grid_size, block, 0, 0>>> (cellpos, pospres, velrhop, idmassidx, str, tensor2d_cache, grid->GetCellBeg(), grid->GetCellEnd());
		getLastCudaError("cuStressRegularizationShepard failed.....     \n");
		checkCudaErrors(cudaMemcpy(str + par_host->npb, tensor2d_cache + par_host->npb, par_host->npm * sizeof(tensor2d), cudaMemcpyDeviceToDevice));
	}
}

void LoquatSph::StressSavingRegularization() {
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));
	dim3 block(128, 1);
	if (par_host->npm != 0) {
		dim3 cuda_grid_size = dim3((par_host->npm + block.x - 1) / block.x, 1);
		checkCudaErrors(cudaMemcpy(tensor2d_cache, str, (par_host->npb + par_host->npm) * sizeof(tensor2d), cudaMemcpyDeviceToDevice));
		cuStressRegularizationShepard << <cuda_grid_size, block, 0, 0 >> > (cellpos, pospres, velrhop, idmassidx, str, tensor2d_cache, grid->GetCellBeg(), grid->GetCellEnd());
		getLastCudaError("cuStressRegularizationShepard failed.....     \n");
	}
}