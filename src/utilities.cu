/*********************************************************************************
 * Copyright (c) 2019, Chong Peng. All rights reserved.

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
 #include "utilities.h"
 
 static __constant__ Parameters par;

__inline__ __device__ void cuGetSurfaceLoop(
	const int part_beg,
	const int part_end,
	const scalar x,
	const scalar y,
	const scalar z,
	scalar4* pospres,
	scalar& max_z)
{
	for(int j = part_beg; j < part_end; j++)
	{
		scalar4 pospres_j = pospres[j];

		scalar3 dr = make_scalar3(x - pospres_j.x, y - pospres_j.y, z - pospres_j.z);
		scalar r = norm3df(dr.x, dr.y, 0.0f);

		if(r < 0.5f * par.dr)
		{
			if(pospres_j.z > max_z) max_z = pospres_j.z;
		}
	}
}


__global__ void cuGetSurface(
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	const int* cell_beg,
	const int* cell_end,
	scalar4* posh,
	int nip){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < nip){
		scalar x = 2.0f * par.h + scalar(index) * (0.5f * par.dr);
		scalar y = 2.0f * par.h + 0.5f * par.domain_size.y;
		scalar z = 2.0f * par.h;
		scalar max_z = 0.0f;
		while(z < par.domain_size.z - 2.0f * par.h){
			z += par.dr;
			int3 celli = make_int3(floorf(0.5f * x * par.i_h),
								   floorf(0.5f * y * par.i_h),
								   floorf(0.5f * z * par.i_h));
		
			for(int cz = -1; cz <= 1; cz++){
				for(int cy = -1; cy <= 1; cy++){
					for(int cx = -1; cx <= 1; cx++){
						int3 cellj = make_int3(celli.x + cx, celli.y + cy, celli.z + cz);

						if(cellj.x < 0 || cellj.x > par.cell_num.x - 1 ||
						   cellj.y < 0 || cellj.y > par.cell_num.y - 1 ||
						   cellj.z < 0 || cellj.z > par.cell_num.z - 1)
							continue;

						int cell_idx = cellj.x * par.cell_num.y * par.cell_num.z + cellj.y * par.cell_num.z + cellj.z;
						cell_idx += par.ncell_material;
						cuGetSurfaceLoop(cell_beg[cell_idx], cell_end[cell_idx], x, y, z, pospres, max_z);
					}
				}
			}
		}

		posh[index] = make_scalar4(x, y, max_z, 0.0f);
	}
}


 void GetSurface(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	const int* cell_beg,
	const int* cell_end)
{
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	int nip = int(2.0f * par_host->domain_size.x / par_host->dr) + 1;
	scalar4* posh_host;
	scalar4* posh_device;
	cudaHostAlloc((void **)&posh_host, nip * sizeof(scalar4), cudaHostAllocDefault);
	cudaMalloc((void **)&posh_device, nip * sizeof(scalar4));
	
	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((nip + block.x - 1) / block.x, 1);

	cuGetSurface <<<cuda_grid_size, block, 0, 0>>> (pospres, velrhop, idmassidx, cell_beg, cell_end, posh_device, nip);

	cudaMemcpy(posh_host, posh_device, nip * sizeof(scalar4), cudaMemcpyDeviceToHost);
	
	std::stringstream ss;
	ss << project_name << "/surface_" << par_host->output_number << ".dat";
	std::string filename = ss.str();

	std::ofstream ofs(filename.c_str(), std::ofstream::out);
	if(ofs)
	{
		for(int i = 0; i < nip; i++)
		{
			ofs << posh_host[i].x - 2.0f * par_host->h << " " << posh_host[i].z - 2.0f * par_host->h + 0.5f * par_host->dr << std::endl;
		}
	}
	ofs.close();
}
 /*
__inline__ __device__ void cuGetvalueAlongLineMaterialLoop(
	const int part_beg,
	const int part_end,
	const scalar x,
	const scalar y,
	const scalar z,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2d* str,
	scalar4& results,
	scalar& sumker_material,
	bool bound)
{
	for(int j = part_beg; j < part_end; j++)
	{
		scalar4 pospres_j = pospres[j];

		scalar3 dr = make_scalar3(x - pospres_j.x, y - pospres_j.y, z - pospres_j.z);
		scalar r = norm3df(dr.x, dr.y, dr.z);
		scalar q = r * par.i_h;
		if(q < 2.0f)
		{
			scalar ker = par.kernel_normalization_par * (2.0f - q) * (2.0f - q) * (2.0f - q) * (2.0f - q) * (q + 0.5f);
	
			scalar vol_j = bound ? par.vol_def : idmassidx[j].y / velrhop[j].w;
			results.w += str[j].zz * ker * vol_j;
			sumker_material += ker * vol_j;
		}
	}
}

__global__ void cuGetvalueAlongLine(
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2d* str,
	tensor2d* stn,
	const int* cell_beg,
	const int* cell_end,
	scalar4* results_device,
	int nip)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < 1.f / par.dr)
	{
		scalar x = 0.5 * par.domain_size.x;
		scalar y = par.dim==TwoD ? 0.f : 0.5 * par.domain_size.y;
		scalar z = scalar(index) * par.dr + par.bound_thickness;

		int3 celli = make_int3(floorf(0.5f * x * par.i_h),
							   floorf(0.5f * y * par.i_h),
							   floorf(0.5f * z * par.i_h));
		
		scalar4 results = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);
		scalar sumker_material = 0.0f;
		for(int cz = -1; cz <= 1; cz++)
		{
			for(int cy = -1; cy <= 1; cy++)
			{
				for(int cx = -1; cx <= 1; cx++)
				{
					int3 cellj = make_int3(celli.x + cx, celli.y + cy, celli.z + cz);

					if(cellj.x < 0 || cellj.x > par.cell_num.x - 1 ||
					   cellj.y < 0 || cellj.y > par.cell_num.y - 1 ||
					   cellj.z < 0 || cellj.z > par.cell_num.z - 1)
					{
						continue;
					}

					int cell_idx = cellj.x * par.cell_num.y * par.cell_num.z + cellj.y * par.cell_num.z + cellj.z;
					
					int cell_idx_material = cell_idx + par.ncell_material;
			//		cuGetvalueAlongLineMaterialLoop(cell_beg[cell_idx], cell_end[cell_idx], x, y, z, pospres, velrhop, epphi, idmassidx, str, results, sumker_material, true);
					cuGetvalueAlongLineMaterialLoop(cell_beg[cell_idx_material], cell_end[cell_idx_material], x, y, z, pospres, velrhop, idmassidx, str, results, sumker_material, false);
				}
			}
		}
		
		if(sumker_material != 0.0f) results.w /= sumker_material;

		results.x = z;

		results_device[index] = results;
	}
}


void GetValuesAlongLine(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2d* str,
	tensor2d* stn,
	const int* cell_beg,
	const int* cell_end)
{
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	int nip = 1.f / par_host->dr;
	scalar4* results_host;
	scalar4* results_device;
	cudaHostAlloc((void **)&results_host, nip * sizeof(scalar4), cudaHostAllocDefault);
	cudaMalloc((void **)&results_device, nip * sizeof(scalar4));
	
	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((nip + block.x - 1) / block.x, 1);

	cuGetvalueAlongLine <<<cuda_grid_size, block, 0, 0>>> (pospres, velrhop, idmassidx, str, stn, cell_beg, cell_end, results_device, nip);

	cudaMemcpy(results_host, results_device, nip * sizeof(scalar4), cudaMemcpyDeviceToHost);
	
	std::stringstream ss;
	ss << project_name << "/stress&pressure" << ".txt";
	std::string filename = ss.str();

	std::ofstream ofs(filename.c_str(), std::ofstream::out);
	if(ofs)
	{
		for(int i = 0; i < nip; i++)
		{
			ofs << results_host[i].x - par_host->bound_thickness << " " << results_host[i].w << std::endl;
		}
	}
	ofs.close();
}*/

__global__ void cuGetValuesAlongLine(
	scalar4* pospres,
	int4* cellpos,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2dful* contact_mag,
	scalar4* contact_var,
	tensor2d* str,
	tensor2d* stn,
	const int* cell_beg,
	const int* cell_end,
	tensor2d* results_device,
	int nip)
{
	for (int i = par.npb; i < par.np; i++) {
		scalar3 pos_abs_i;
		pos_abs_i.x = pospres[i].x + cellpos[i].x * (2.0 * par.h);
		pos_abs_i.y = pospres[i].y + cellpos[i].y * (2.0 * par.h);
		pos_abs_i.z = pospres[i].z + cellpos[i].z * (2.0 * par.h);
		bool bound_move1 = (pos_abs_i.x >= 0.04f + par.bound_thickness - par.time * 0.005 + par.dr * 0)//three layers
			&& (pos_abs_i.x <= 0.04f + par.bound_thickness - par.time * 0.005 + par.dr * 1)
			&& (pos_abs_i.z >= par.bound_thickness + 0.04f)
			&& (pos_abs_i.z <= par.bound_thickness + 0.04f + 1.0f);
		bool bound_move2 = (pos_abs_i.x >= 0.04f + par.bound_thickness - par.time * 0.005 + par.dr * 1)
			&& (pos_abs_i.x <= 0.04f + par.bound_thickness - par.time * 0.005 + par.dr * 2)
			&& (pos_abs_i.z >= par.bound_thickness + 0.04f)
			&& (pos_abs_i.z <= par.bound_thickness + 0.04f + 1.0f);
		bool bound_move3 = (pos_abs_i.x >= 0.04f + par.bound_thickness - par.time * 0.005 + par.dr * 2)
			&& (pos_abs_i.x <= 0.04f + par.bound_thickness - par.time * 0.005 + par.dr * 3)
			&& (pos_abs_i.z >= par.bound_thickness + 0.04f)
			&& (pos_abs_i.z <= par.bound_thickness + 0.04f + 1.0f);
		if (bound_move1) {
			int num_z = (0.04f + 1.0f + par.bound_thickness - pos_abs_i.z) / par.dr;
			results_device[num_z].xy = num_z * par.dr;//depth from 0
			results_device[num_z].xx = contact_mag[i].xx / par.dr;// str[i].zz;//pressure(Pa)
		}
		if (bound_move2) {
			int num_z = (0.04f + 1.0f + par.bound_thickness - pos_abs_i.z) / par.dr;
			results_device[num_z].yy = contact_mag[i].xx / par.dr;// str[i].zz;//pressure(Pa)
		}
		if (bound_move3) {
			int num_z = (0.04f + 1.0f + par.bound_thickness - pos_abs_i.z) / par.dr;
			results_device[num_z].zz = contact_mag[i].xx / par.dr;// str[i].zz;//pressure(Pa)
		}

		/*// x_mesh = (int)(pos_abs_i.x * 1000) % (int)(par.dr * 5 * 1000) < 1.5 * par.dr * 1000;//*1000 to be int
		//bool y_mesh = (int)(pos_abs_i.y * 1000) % (int)(par.dr * 5 * 1000) < 1.5 * par.dr * 1000;
		int num_x = pos_abs_i.x / (par.dr * 5);
		int num_z = pos_abs_i.z / (par.dr * 5);
		results_device[num_x*40 + num_z].xx = pos_abs_i.x;
		results_device[num_x*40 + num_z].zz = pos_abs_i.z;
		results_device[num_x*40 + num_z].xy = str[i].xx;
		results_device[num_x*40 + num_z].xz = str[i].zz;
		results_device[num_x*40 + num_z].yz = str[i].xz;*/
	}
}


void GetValuesAlongLine(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2dful* contact_mag,
	scalar4* contact_var,
	tensor2d* str,
	tensor2d* stn,
	int4* cellpos,
	const int* cell_beg,
	const int* cell_end)
{
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	int nip = (1.0f) / par_host->dr;
	tensor2d* results_host;
	tensor2d* results_device;
	cudaHostAlloc((void**)&results_host, nip * sizeof(tensor2d), cudaHostAllocDefault);
	cudaMalloc((void**)&results_device, nip * sizeof(tensor2d));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((nip + block.x - 1) / block.x, 1);

	cuGetValuesAlongLine << <cuda_grid_size, block, 0, 0 >> > (pospres, cellpos, velrhop, idmassidx, contact_mag, contact_var, str, stn, cell_beg, cell_end, results_device, nip);

	cudaMemcpy(results_host, results_device, nip * sizeof(tensor2d), cudaMemcpyDeviceToHost);

	std::stringstream ss;
	ss << project_name << "/stress_zz" << par_host->output_number << ".txt";
	std::string filename = ss.str();

	std::ofstream ofs(filename.c_str(), std::ofstream::out);
	if (ofs)
	{
		for (int i = 0; i < nip; i++)
		{
			ofs << results_host[i].xy << " " << results_host[i].xx << " " << results_host[i].yy << " " << results_host[i].zz << std::endl;
		}
	}
	ofs.close();
}



__inline__ __device__ void cuGetSurfaceAtPointLoop(
	const int part_beg,
	const int part_end,
	scalar x,
	scalar y,
	scalar z,
	scalar4* pospres,
	scalar& height)
{
	for(int i = part_beg; i < part_end; i++)
	{
		scalar4 pospres_i = pospres[i];
		if(fabsf(x - pospres_i.x) < 2.0f * par.h)
		{
			if(pospres_i.z > height) height = pospres_i.z;
		}
	}
}


__global__ void cuGetSurfaceAtPoint(
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	const int* cell_beg,
	const int* cell_end,
	scalar4* results_device)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < 2)
	{
		scalar x = 2.0f * par.h + 0.4f * scalar(index + 1);
		scalar y = 0.0f;
		scalar z = 2.0f * par.h;	

		scalar height = -1.0e10f;

		int3 celli = make_int3(floorf(0.5f * x * par.i_h),
							   floorf(0.5f * y * par.i_h),
							   floorf(0.5f * z * par.i_h));
	
		for(int cz = 0; cz < par.cell_num.z; cz++)
		{
			for(int cx = -1; cx <= 1; cx++)
			{
				int3 cellj = make_int3(celli.x + cx, celli.y, cz);
				
				if(cellj.x < 0 || cellj.x > par.cell_num.x - 1 ||
				   cellj.y < 0 || cellj.y > par.cell_num.y - 1 ||
				   cellj.z < 0 || cellj.z > par.cell_num.z - 1)
				 {
					 continue;
				 }

				 int cell_idx = cellj.x * par.cell_num.y * par.cell_num.z + cellj.y * par.cell_num.z + cellj.z;
				cell_idx += par.ncell_material;

				cuGetSurfaceAtPointLoop(cell_beg[cell_idx], cell_end[cell_idx], x, y, z, pospres, height);
			}
		}

		results_device[index] = make_scalar4(x, y, z, height);
	}
}

void GetSurfaceAtPoint(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	const int* cell_beg,
	const int* cell_end,
	std::ofstream& time_history_ofs)
{
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	scalar4* results_host;
	scalar4* results_device;
	cudaHostAlloc((void **)&results_host, 2 * sizeof(scalar4), cudaHostAllocDefault);
	cudaMalloc((void **)&results_device, 2 * sizeof(scalar4));

	cuGetSurfaceAtPoint <<<1, 2, 0, 0>>> (pospres, velrhop, idmassidx, cell_beg, cell_end, results_device);

	cudaMemcpy(results_host, results_device, 2 * sizeof(scalar4), cudaMemcpyDeviceToHost);

	if(time_history_ofs)
	{
		time_history_ofs << par_host->time << " " 
						 << results_host[0].x - 2.0f * par_host->h << " " 
						 << results_host[0].w - 2.0f * par_host->h + 0.5f * par_host->dr << " "
						 << results_host[1].x - 2.0f * par_host->h << " "
						 << results_host[1].w - 2.0f * par_host->h + 0.5f * par_host->dr
						 << std::endl;
	}
}

__inline__ __device__ void cuGetStressZz(
	const int part_beg,
	const int part_end,
	const scalar x,
	const scalar y,
	const scalar z,
	scalar4* pospres,
	int4* cellpos,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2d* str,
	scalar4& results)
{
	for(int j = part_beg; j < part_end; j++)
	{
		scalar3 pos_abs_j;//Transition from relative position to absolute position
		pos_abs_j.x = pospres[j].x + cellpos[j].x * (2.0f * par.h);
		pos_abs_j.y = pospres[j].y + cellpos[j].y * (2.0f * par.h);
		pos_abs_j.z = pospres[j].z + cellpos[j].z * (2.0f * par.h);
		scalar3 dr = make_scalar3(x - pos_abs_j.x, y - pos_abs_j.y, z - pos_abs_j.z);
		scalar r = norm3df(dr.x, dr.y, dr.z);
		scalar q = r * par.i_h;
		if(q < 2.0f)
		{
			scalar ker = GetKernel(par.kernel_normalization_par, q);
			
			scalar vol_j = idmassidx[j].y / velrhop[j].w;
			results.z += str[j].xx * ker * vol_j;
			results.w += ker * vol_j;
		}
	}
}


__global__ void GetSoilStress(
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2dful* contact_mag,
	scalar4* contact_var,
	tensor2d* str,
	int4* cellpos,
	const int* cell_beg,
	const int* cell_end,
	scalar4* results_device)
{
	scalar x = 0.1f + par.bound_thickness + par.time * 0.1f + par.dr * 0;
	scalar y = par.dim == TwoD ? 0.0f : 0.5f * par.domain_size.y;
	scalar z = 0.1f + par.bound_thickness + 0.12f;//height of sensor 0.03 0.12 0.21 0.3 0.39 0.48
	
	int3 celli = make_int3(floorf(0.5f * x * par.i_h),
						   floorf(0.5f * y * par.i_h),
						   floorf(0.5f * z * par.i_h));
	
	scalar4 results = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);

	for(int cz = 0; cz < par.cell_num.z; cz++)
	{
		for(int cx = -1; cx <= 1; cx++)
		{
			int3 cellj = make_int3(celli.x + cx, celli.y, cz);
				
			if(cellj.x < 0 || cellj.x > par.cell_num.x - 1 ||
			   cellj.y < 0 || cellj.y > par.cell_num.y - 1 ||
			   cellj.z < 0 || cellj.z > par.cell_num.z - 1)
			 {
				 continue;
			 }

			int cell_idx = cellj.x * par.cell_num.y * par.cell_num.z + cellj.y * par.cell_num.z + cellj.z;
			int cell_idx_material = cell_idx + par.ncell_material;
			cuGetStressZz(cell_beg[cell_idx_material], cell_end[cell_idx_material], x, y, z, pospres, cellpos, velrhop, idmassidx, str, results);
		}
	}
	if(results.w != 0.0f) results.z /= results.w;

	results_device[0] = results;
}


void InterpolateResults(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2dful* contact_mag,
	scalar4* contact_var,
	tensor2d* str,
	int4* cellpos,
	const int* cell_beg,
	const int* cell_end,
	std::ofstream& time_history_ofs)
{
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	int nip = 1;
	scalar4* results_device;
	cudaMalloc((void **)&results_device, nip * sizeof(scalar4));
	GetSoilStress <<<1, 1, 0, 0>>> (pospres, velrhop, idmassidx, contact_mag, contact_var, str, cellpos, cell_beg, cell_end, results_device);

	scalar4* results_host;
	cudaHostAlloc((void **)&results_host, nip * sizeof(scalar4), cudaHostAllocDefault);
	cudaMemcpy(results_host, results_device, nip * sizeof(scalar4), cudaMemcpyDeviceToHost);
	
	time_history_ofs << par_host->time << " " << results_host[0].z << std::endl;
}

__global__ void cuGetValuesAtPoint(
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2dful* contact_mag,
	scalar4* contact_var,
	tensor2d* str,
	int4* cellpos,
	const int* cell_beg,
	const int* cell_end,
	scalar4* results_device)
{
	scalar4 results = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);
	
	for (int i = par.npb; i < par.np; i++) {
		scalar3 pos_abs_i;
		pos_abs_i.x = pospres[i].x + cellpos[i].x * (2.0f * par.h);
		pos_abs_i.y = pospres[i].y + cellpos[i].y * (2.0f * par.h);
		pos_abs_i.z = pospres[i].z + cellpos[i].z * (2.0f * par.h);
		
		/*bool SPT_2 = (pos_abs_i.x >= 0.05)
			&& (pos_abs_i.x <= 0.05 + 0.42 - par.h * 3)
			&& (pos_abs_i.z >= 0.05 + par.bound_thickness + 0.12 - par.h)
			&& (pos_abs_i.z <= 0.05 + par.bound_thickness + 0.12 + par.h);
		bool SPT_4 = (pos_abs_i.x >= 0.05)
			&& (pos_abs_i.x <= 0.05 + 0.42 - par.h * 3)
			&& (pos_abs_i.z >= 0.05 + par.bound_thickness + 0.3 - par.h)
			&& (pos_abs_i.z <= 0.05 + par.bound_thickness + 0.3 + par.h);
		bool SPT_6 = (pos_abs_i.x >= 0.05)
			&& (pos_abs_i.x <= 0.05 + 0.42 - par.h * 3)
			&& (pos_abs_i.z >= 0.05 + par.bound_thickness + 0.48 - par.h)
			&& (pos_abs_i.z <= 0.05 + par.bound_thickness + 0.48 + par.h);
		bool SPT_8 = (pos_abs_i.x >= 0.05)
			&& (pos_abs_i.x <= 0.05 + 0.42 - par.h * 3)
			&& (pos_abs_i.z >= 0.05 + par.bound_thickness + 0.66 - par.h)
			&& (pos_abs_i.z <= 0.05 + par.bound_thickness + 0.66 + par.h);
		
		bool All = (pos_abs_i.x >= 0.04f + par.bound_thickness - par.time * 0.005 + par.dr * 0)//three layers
			&& (pos_abs_i.x <= 0.04f + par.bound_thickness - par.time * 0.005 + par.dr * 3)
			&& (pos_abs_i.z >= par.bound_thickness + 0.04f)
			&& (pos_abs_i.z <= par.bound_thickness + 0.04f + 1.0f);*/
		bool All = (pos_abs_i.x <= 0.04f + par.bound_thickness + par.dr * 3)
			&& (pos_abs_i.z >= par.bound_thickness + 0.04f)
			&& (pos_abs_i.z <= par.bound_thickness + 0.04f + 1.0f);

		if (All) {
			results.x += 0.001 * contact_mag[i].xx;//output kN
		}
		/*if (SPT_2) {
			results.x += 0.001 * contact_mag[i].xx / (2 * par.h);//output kPa
		}
		else if (SPT_4) {
			results.y += 0.001 * contact_mag[i].xx / (2 * par.h);//output kPa
		}
		else if (SPT_6) {
			results.z += 0.001 * contact_mag[i].xx / (2 * par.h);//output kPa
		}
		else if (SPT_8) {
			results.w += 0.001 * contact_mag[i].xx / (2 * par.h);//output kPa
		}*/
	}
	results_device[0] = results;
}


void GetValuesAtPoint(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2dful* contact_mag,
	scalar4* contact_var,
	tensor2d* str,
	int4* cellpos,
	const int* cell_beg,
	const int* cell_end,
	std::ofstream& time_history_ofs)
{
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	int nip = 1;
	scalar4* results_device;
	cudaMalloc((void**)&results_device, nip * sizeof(scalar4));
	cuGetValuesAtPoint << <1, 1, 0, 0 >> > (pospres, velrhop, idmassidx, contact_mag, contact_var, str, cellpos, cell_beg, cell_end, results_device);

	scalar4* results_host;
	cudaHostAlloc((void**)&results_host, nip * sizeof(scalar4), cudaHostAllocDefault);
	cudaMemcpy(results_host, results_device, nip * sizeof(scalar4), cudaMemcpyDeviceToHost);

	time_history_ofs << par_host->time << " " << results_host[0].x << std::endl;
}