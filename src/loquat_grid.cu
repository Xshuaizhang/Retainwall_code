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

#include <math.h>
//#include <cuda.h>
#include <helper_cuda.h>
#include <cub/cub.cuh>
#include "loquat_grid.h"

static __constant__ Parameters par;

LoquatGrid::LoquatGrid()
{
	part_cel = NULL;
    part_idx = NULL;
    cell_beg = NULL;
    cell_end = NULL;

	part_cel_cache = NULL;
	part_idx_cache = NULL;
	buffer = NULL;

	buffer_storage_bytes = 0;
}


LoquatGrid::~LoquatGrid()
{
    if(part_cel != NULL) cudaFree(part_cel);
    if(part_idx != NULL) cudaFree(part_idx);
    if(cell_beg != NULL) cudaFree(cell_beg);
    if(cell_end != NULL) cudaFree(cell_end);

	if(part_cel_cache != NULL) cudaFree(part_cel_cache);
	if(part_idx_cache != NULL) cudaFree(part_idx_cache);
	if(buffer != NULL) cudaFree(buffer);

	buffer_storage_bytes = 0;
}


void LoquatGrid::GridConfig(Parameters* par_host)
{
	checkCudaErrors(cudaMalloc((void **)&part_cel, par_host->np * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&part_idx, par_host->np * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&cell_beg, par_host->ncell * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&cell_end, par_host->ncell * sizeof(int)));

	checkCudaErrors(cudaMalloc((void **)&part_cel_cache, par_host->np * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&part_idx_cache, par_host->np * sizeof(int)));
}


__global__ void cuGetParticleCellIndex(
		int* part_cel,
		int* part_idx,
		int4* cellpos,
		scalar4* idmassidx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i < par.np)
	{
		if(cellpos[i].w == par.ncell_delete_m){
			part_cel[i] = par.ncell_delete_m;
			part_idx[i] = i;
			return;
		}

		int4 celli = cellpos[i];
		int cell_idx = celli.x * par.cell_num.y * par.cell_num.z + celli.y * par.cell_num.z + celli.z;
		if (idmassidx[i].x < MATERIAL_NUMBER) part_cel[i] = cell_idx; // Boundary		
		else if (idmassidx[i].x >= MATERIAL_NUMBER) part_cel[i] = cell_idx + par.ncell_material; // Material		

		cellpos[i].w = part_cel[i];
		part_idx[i] = i;
	}
}


void LoquatGrid::SortParticleByCellIndex(Parameters* par_host){
	size_t temp_storage_bytes = 0;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, part_cel, part_cel_cache, part_idx, part_idx_cache, par_host->np));

	if(temp_storage_bytes > buffer_storage_bytes){
		buffer_storage_bytes = temp_storage_bytes;
		if(buffer != NULL) cudaFree(buffer);
		checkCudaErrors(cudaMalloc(&buffer, buffer_storage_bytes));
	}

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(buffer, temp_storage_bytes, part_cel, part_cel_cache, part_idx, part_idx_cache, par_host->np));

	checkCudaErrors(cudaMemcpyAsync(part_cel, part_cel_cache, par_host->np * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(part_idx, part_idx_cache, par_host->np * sizeof(int), cudaMemcpyDeviceToDevice));
}


__global__ void cuFindBegEndInCell(int* cell_beg, int* cell_end, int* part_cel)
{
	extern __shared__ int shared_cells[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int celli;

	if(i < par.np)
	{
		celli = part_cel[i];
		shared_cells[threadIdx.x + 1] = celli;

		// First thread in this block
		if(i > 0 && threadIdx.x == 0) shared_cells[0] = part_cel[i - 1];
	}

	__syncthreads();

	if(i < par.np)
	{
		if(i == 0 || celli != shared_cells[threadIdx.x])
		{
			cell_beg[celli] = i;
			if(i > 0) cell_end[shared_cells[threadIdx.x]] = i;
		}

		if(i == par.np - 1) cell_end[celli] = i + 1;
	}

}


void LoquatGrid::NeighborSearch(Parameters* par_host, Parameters* par_device, int4* cellpos, scalar4* idmassidx)
{
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->np + block.x - 1) / block.x, 1);

	cuGetParticleCellIndex <<<cuda_grid_size, block, 0>>> (part_cel, part_idx, cellpos, idmassidx);
	getLastCudaError("cuGetParticleCellIndex failed...     \n");

	SortParticleByCellIndex(par_host);

	checkCudaErrors(cudaMemsetAsync(cell_beg, -1, par_host->ncell * sizeof(int)));
	checkCudaErrors(cudaMemsetAsync(cell_end, -1, par_host->ncell * sizeof(int)));

	cuFindBegEndInCell <<<cuda_grid_size, block, (block.x + 1) * sizeof(int)>>> (cell_beg, cell_end, part_cel);
	getLastCudaError("cuFindBegEndInCell failed...     \n");

	int beg = 0;
	int np_delete_m_now = 0;
	checkCudaErrors(cudaMemcpy(&beg, cell_beg + par_host->ncell_delete_m, sizeof(int), cudaMemcpyDeviceToHost));
	if(beg != -1)
	{
		int end = 0;
		checkCudaErrors(cudaMemcpy(&end, cell_end + par_host->ncell_delete_m, sizeof(int), cudaMemcpyDeviceToHost));
		np_delete_m_now = end - beg - par_host->np_delete_m;
		if(np_delete_m_now != 0)
		{
			par_host->npm -= np_delete_m_now;
			par_host->np_delete_m += np_delete_m_now;

			printf("Deleting material particle: %d, total deleted material particle: %d \n", np_delete_m_now, par_host->np_delete_m);
		}
	}

	if(par_host->npb + par_host->npm + par_host->np_delete_m != par_host->np){
		printf("ERROR in deleting particles ... \n");
		exit(EXIT_FAILURE);
	}

	if(np_delete_m_now != 0){
		checkCudaErrors(cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));
	}
}

static __global__ void cuSortArrayInt4(int size, int4* data, int4* data_cache, int* part_idx) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) data[i] = data_cache[part_idx[i]];
}

void LoquatGrid::SortArrayInt4(int size, int4* data, int4* data_cache) {
	checkCudaErrors(cudaMemcpyAsync(data_cache, data, size * sizeof(int4), cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((size + block.x - 1) / block.x, 1);

	cuSortArrayInt4 << <cuda_grid_size, block, 0 >> > (size, data, data_cache, part_idx);
	getLastCudaError("cuSortArrayFloat4 failed...     \n");
}

static __global__ void cuSortArrayScalar4(int size, scalar4* data, scalar4* data_cache, int* part_idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size)
	{
		data[i] = data_cache[part_idx[i]];
	}
}


void LoquatGrid::SortArrayScalar4(int size, scalar4* data, scalar4* data_cache)
{
	checkCudaErrors(cudaMemcpyAsync(data_cache, data, size * sizeof(scalar4), cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((size + block.x - 1) / block.x, 1);

	cuSortArrayScalar4 <<<cuda_grid_size, block, 0>>> (size, data, data_cache, part_idx);
	getLastCudaError("cuSortArrayScalar4 failed...     \n");
}


static __global__ void cuSortArrayTensor2d(int size, tensor2d* data, tensor2d* data_cache, int* part_idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size)
	{
		data[i] = data_cache[part_idx[i]];
	}
}

void LoquatGrid::SortArrayTensor2d(int size, tensor2d* data, tensor2d* data_cache)
{
	checkCudaErrors(cudaMemcpyAsync(data_cache, data, size * sizeof(tensor2d), cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((size + block.x - 1) / block.x, 1);

	cuSortArrayTensor2d <<<cuda_grid_size, block, 0>>> (size, data, data_cache, part_idx);
	getLastCudaError("cuSortArrayTensor2d failed...     \n");
}

__global__ void cuPositionFromAbsolute2Relative(int4* cellpos, scalar4* pospres) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < par.np) {
		scalar3 pos_abs_i = make_scalar3(pospres[i]);
		int3 cell_pos_i = make_int3(pos_abs_i * (0.5 * par.i_h));
		cellpos[i] = make_int4(cell_pos_i);
		pospres[i] = pospres[i] - cell_pos_i * (2.0 * par.h);
	}
}

void LoquatGrid::PositionFromAbsolute2Relative(Parameters* par_host, Parameters* par_device, int4* cellpos, scalar4* pospres) {
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->np + block.x - 1) / block.x, 1);

	cuPositionFromAbsolute2Relative << <cuda_grid_size, block, 0 >> > (cellpos, pospres);
	getLastCudaError("cuPositionFromAbsolute2Relative failed...     \n");
}

__global__ void cuPositionFromRelative2Absolute(int4* cellpos, scalar4* pospres, scalar4* scalar4_cache) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < par.np) {
		int3 cellpos_i = make_int3(cellpos[i]);
		scalar4_cache[i] = pospres[i] + cellpos_i * (2.0 * par.h);
	}
}

void LoquatGrid::PositionFromRelative2Absolute(Parameters* par_host, Parameters* par_device, int4* cellpos, scalar4* pospres, scalar4* scalar4_cache) {
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->np + block.x - 1) / block.x, 1);

	cuPositionFromRelative2Absolute << <cuda_grid_size, block, 0 >> > (cellpos, pospres, scalar4_cache);
	getLastCudaError("cuPositionFromRelative2Absolute failed...     \n");
}