/*********************************************************************************
 * Copyright (c) 2018, Chong Peng, 2022, Chengwei Zhu. All rights reserved.

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

static __inline__ __device__ void cuAdamiBoundaryMaterialLoopSolution(int part_beg, int part_end, int3 cell_offset, scalar3 pos_i, scalar4* pospres,
	scalar4* velrhop, scalar4* idmassidx, tensor2d* str, tensor2d& results_stress, scalar4& results_pos_sumker, scalar4& results_velrhop, bool contact, scalar3& n_i, const bool bound){
	for(int j = part_beg; j < part_end; j++){
		scalar3 pos_j = make_scalar3(pospres[j]);
		scalar3 dr = pos_i - pos_j;
		dr = dr - cell_offset * (2.0 * par.h);
		scalar r = norm3df(dr.x, dr.y, dr.z);
		scalar q = r * par.i_h;
		if(q < 2.0f) {
			scalar ker = GetKernel(par.kernel_normalization_par, q);
			scalar dwdr = GetGradKernel(par.kernel_gradient_normalization_par, q);
			scalar3 gker = dr * dwdr;
			scalar vol_j = idmassidx[j].y / velrhop[j].w;
			if (bound && contact) {
				n_i = n_i - gker * par.vol_def;// Integrate ker of structure particles
			}
			if (!bound) {
				results_stress = results_stress + str[j] * (ker * vol_j);
				results_pos_sumker = results_pos_sumker + make_scalar4(dr * (ker * idmassidx[j].y), ker * vol_j);
				results_velrhop = results_velrhop + velrhop[j] * (ker * vol_j);
			}
		}
	}
}

static __inline__ __device__ void cuAdamiBoundaryMaterialLoop(const int3 celli, const scalar3 pos_i, scalar4* pospres, scalar4* velrhop,
	scalar4* idmassidx, tensor2d* str, const int* cell_beg, const int* cell_end, tensor2d& results_stress, scalar4& results_pos_sumker, scalar4& results_velrhop, const bool contact, scalar3& n_i) {
	for (int cz = -1; cz <= 1; cz++) {
		for (int cy = -1; cy <= 1; cy++) {
			for (int cx = -1; cx <= 1; cx++) {
				int3 cell_offset = make_int3(cx, cy, cz);
				int3 cellj = celli + cell_offset;
				if ((!(cellj >= make_int3(0))) || (!(cellj < par.cell_num)))continue;
				
				// Interaction with boundary particles
				int cell_idx = cellj.x * par.cell_num.y * par.cell_num.z + cellj.y * par.cell_num.z + cellj.z;
				int cell_idx_bound = cell_idx;
				cuAdamiBoundaryMaterialLoopSolution(cell_beg[cell_idx_bound], cell_end[cell_idx_bound], cell_offset, pos_i, pospres, velrhop, idmassidx, str, results_stress, results_pos_sumker, results_velrhop, contact, n_i, true);
				
				// Interaction with material particles
				int cell_idx_material = cell_idx + par.ncell_material;
				cuAdamiBoundaryMaterialLoopSolution(cell_beg[cell_idx_material], cell_end[cell_idx_material], cell_offset, pos_i, pospres, velrhop, idmassidx, str, results_stress, results_pos_sumker, results_velrhop, contact, n_i, false);
			}
		}
	}
}

static __inline__ __device__ void cuAdamiBoundaryMaterialLoopPeriod(const int3 celli, const scalar3 pos_i, scalar4* pospres, scalar4* velrhop,
scalar4* idmassidx, tensor2d* str, const int* cell_beg, const int* cell_end, tensor2d& results_stress, scalar4& results_pos_sumker, 
scalar4& results_velrhop, const bool contact, scalar3& n_i) {
	scalar3 pos_abs_i = pos_i + celli * (2.0 * par.h); //Transition from relative position to absolute position
	bool period_left = AbsDist2Plane(pos_abs_i, par.planeleft) < 2.0 * par.h;
	bool period_right = AbsDist2Plane(pos_abs_i, par.planeright) < 2.0 * par.h;
	if (period_left || period_right) {
		scalar shift_length = abs(par.planeleft.w + par.planeright.w);
		scalar3 shift_vector = period_left? make_scalar3(par.planeleft) * shift_length : make_scalar3(par.planeright) * shift_length;
		scalar3 pos_abs_pb_i = pos_abs_i + shift_vector;
		int3 cell_pb_i = make_int3(pos_abs_pb_i * (0.5 * par.i_h));
		scalar3 pos_pb_i = pos_abs_pb_i - cell_pb_i * (2.0 * par.h);
		cuAdamiBoundaryMaterialLoop(cell_pb_i, pos_pb_i, pospres, velrhop, idmassidx, str, cell_beg, cell_end, results_stress, results_pos_sumker, results_velrhop, contact, n_i);
	}
}

__global__ void cuAdamiBoundaryMaterial(int4*codes, int4* cellpos, scalar4* pospres, scalar4* velrhop, scalar4* idmassidx, scalar4* normal,
	tensor2d* str, tensor2dful* contact_mag, scalar4* contact_var, const int* cell_beg, const int* cell_end){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < par.npb) {
		scalar3 pos_i = make_scalar3(pospres[i]);	
		tensor2d results_stress = make_tensor2d(0.0);
		scalar4 results_pos_sumker = make_scalar4(0.0);
		scalar4 results_velrhop = make_scalar4(0.0);
		int3 celli = make_int3(cellpos[i]);
		
		scalar3 n_i = make_scalar3(0.0);//integrate normal of Adami
		bool contact = (codes[i].x >= 10);//use contact method
		
		cuAdamiBoundaryMaterialLoop(celli, pos_i, pospres, velrhop, idmassidx, str, cell_beg, cell_end, results_stress, results_pos_sumker, results_velrhop, contact, n_i);
		if (par.periodbound)cuAdamiBoundaryMaterialLoopPeriod(celli, pos_i, pospres, velrhop, idmassidx, str, cell_beg, cell_end, results_stress, results_pos_sumker, results_velrhop, contact, n_i);

		//transfer normal of Adami
		if (EulerianNorm(n_i) > 0) {
			contact_mag[i].yx = n_i.x / EulerianNorm(n_i);
			contact_mag[i].yy = n_i.y / EulerianNorm(n_i);
			contact_mag[i].yz = n_i.z / EulerianNorm(n_i);
		}

		if(results_pos_sumker.w > 0.0f){
			scalar kersum = results_pos_sumker.w;
			scalar3 dr_ref = make_scalar3(results_pos_sumker);
			scalar friction_coef = normal[i].w; //1: free-slip -1: nonslip
			scalar3 normal_i = make_scalar3(normal[i]); // normal vector unit
			
			// velocity of reference point
			scalar3 vel_ref = make_scalar3(results_velrhop) / kersum;	// velocity at the reference point
			scalar3 vel_ref_n = normal_i * DotProduct(normal_i, vel_ref); // normal velocity at the reference point
			scalar3 vel_ref_t = vel_ref - vel_ref_n; // tangential velocity at the reference point
			
			// velocity of rigid body
			scalar3 vel_rb = make_scalar3(0.0f);      
			if (codes[i].y == NonFix) {
				vel_rb = par.mv[codes[i].w].trans_vel;
				if (par.mv[codes[i].w].is_rotat) {
					scalar3 pos_abs_i = pos_i + celli * (2.0f * par.h);
					scalar3 pos_abs_diff_i_rp = pos_abs_i - par.mv[codes[i].w].pos_rp;
					vel_rb = vel_rb + CrossProduct(par.mv[codes[i].w].rotat_vel, pos_abs_diff_i_rp);
				}
			}
			scalar3 vel_rb_n = normal_i * DotProduct(normal_i, vel_rb);
			scalar3 vel_rb_t = vel_rb - vel_rb_n;

			// extrapolated velocity at the i-th structure point
			scalar3 vel_ext_n_i = vel_rb_n * 2.0f - vel_ref_n;
			scalar3 vel_ext_t_i = friction_coef == -1.0f ? vel_rb_t * 2.0f - vel_ref_t : vel_ref_t;
			scalar3 vel_ext_i = vel_ext_n_i + vel_ext_t_i;
			
			velrhop[i] = make_scalar4(vel_ext_i, results_velrhop.w / kersum);
			//velrhop[i].w = results_velrhop.w / kersum;
			if (contact) {
				velrhop[i] = make_scalar4(vel_rb, results_velrhop.w / kersum);//pre-determined Vel of contact particles
				contact_mag[i].zx = vel_ext_i.x;//extrapolated Vel of contact particles
				contact_mag[i].zy = vel_ext_i.y;
				contact_mag[i].zz = vel_ext_i.z;
			}

			tensor2d str_ref = results_stress / kersum; // stress at the reference point
			str[i] = str_ref - par.gravity * dr_ref / kersum;
		}
		else
		{
			str[i] = make_tensor2d(0.0);
		}
	}
}

void LoquatSph::AdamiBoundary() {
	if(par_host->npb == 0) return;
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));
	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->npb + block.x - 1) / block.x, 1);
	if(par_host->npm != 0){
		cuAdamiBoundaryMaterial <<<cuda_grid_size, block, 0, 0>>> (codes, cellpos, pospres, velrhop, idmassidx, normal_m, str, contact_mag, contact_var, grid->GetCellBeg(), grid->GetCellEnd());
		getLastCudaError("cuAdamiBoundaryMaterial failed.....     \n");
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LoquatSph::RigidBodyUpdate() {
	for (int i = 0; i < par_host->n_motion; i++) {
		if (io->MotionInfoALL[i].is_rotation) {
			for (int j = 0; j < io->MotionInfoALL[i].table_size_rotat; j++) {
				if (par_host->time < io->MotionInfoALL[i].rotat_time[j]) {
					io->MotionInfoALL[i].rotat_vel = io->MotionInfoALL[i].rotat_vel + io->MotionInfoALL[i].rotat_acc[j] * par_host->dt;
					scalar3 vel = io->MotionInfoALL[i].trans_vel[j] + io->MotionInfoALL[i].trans_acc[j] * (0.5 * par_host->dt);
					io->MotionInfoALL[i].pos_rp = io->MotionInfoALL[i].pos_rp + vel * par_host->dt;
					par_host->mv[i].rotat_acc = io->MotionInfoALL[i].rotat_acc[j];
					par_host->mv[i].rotat_vel = io->MotionInfoALL[i].rotat_vel;
					par_host->mv[i].pos_rp = io->MotionInfoALL[i].pos_rp;
					break;
				}
			}
		}
		if (io->MotionInfoALL[i].is_translation) {
			for (int j = 0; j < io->MotionInfoALL[i].table_size_trans; j++) {
				if (par_host->time < io->MotionInfoALL[i].trans_time[j]) {
					io->MotionInfoALL[i].trans_vel[j] = io->MotionInfoALL[i].trans_vel[j] + io->MotionInfoALL[i].trans_acc[j] * par_host->dt;
					par_host->mv[i].trans_acc = io->MotionInfoALL[i].trans_acc[j];
					par_host->mv[i].trans_vel = io->MotionInfoALL[i].trans_vel[j];
					break;
				}
			}
		}		
	}
	
	checkCudaErrors(cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static __inline__ __device__ void cuFindCornerParticleLoop(int part_beg, int part_end, scalar3 pos_i, int3 cell_offset, int4* codes, int4* cellpos, scalar4* pospres,
	scalar4* velrhop, scalar4* idmassidx, const tensor2d str_over_rho2_i, scalar3& acc_s_i, tensor2d* str, tensor2dful* contact_mag, scalar4* contact_var, scalar& d_ij, const int i, bool bound) {
	for (int j = part_beg; j < part_end; j++) {

		scalar3 pos_j = make_scalar3(pospres[j]);
		scalar3 dr = pos_i - pos_j - cell_offset * (2.0 * par.h);
		scalar r = norm3df(dr.x, dr.y, dr.z);
		scalar q = r * par.i_h;

		if (q < 2.0) {
			scalar ker = GetKernel(par.kernel_normalization_par, q);
			scalar dwdr = GetGradKernel(par.kernel_gradient_normalization_par, q);
			scalar3 gker = dr * dwdr;

			if (codes[j].x >= 10) {
				if (r < d_ij && r > 0) {
					scalar3 normal_adami = make_scalar3(contact_mag[j].yx, contact_mag[j].yy, contact_mag[j].yz);
					scalar3 normal_material = dr / r;

					d_ij = r;//find the nearest particle j
					contact_var[i].y = abs(DotProduct(dr, normal_adami));//Distance to the nearest particle j along ni
					contact_var[i].w = codes[j].x; //Contact structure ID

					//Vel of the nearest particle j of i
					contact_mag[i].zx = velrhop[j].x;
					contact_mag[i].zy = velrhop[j].y;
					contact_mag[i].zz = velrhop[j].z;

					scalar cos_corner = 0.866f;// cos(pi / 6)
					scalar cos_ni_nj = abs(DotProduct(normal_material, normal_adami));//cos between ni and nj
					if (cos_ni_nj < cos_corner) {
						//Corner: ni of material = the direction in which the two particles are connected
						contact_mag[i].yx = normal_material.x;
						contact_mag[i].yy = normal_material.y;
						contact_mag[i].yz = normal_material.z;
					}
					else {
						//Else: ni of material = ni of closest adami particle
						contact_mag[i].yx = normal_adami.x;
						contact_mag[i].yy = normal_adami.y;
						contact_mag[i].yz = normal_adami.z;
					}

				}
				// Integrate ker of external
				contact_var[i].x += abs((contact_mag[i].yx * gker.x + contact_mag[i].yy * gker.y + contact_mag[i].yz * gker.z) * par.vol_def);
				
			}
		}
	}
}

__global__ void cuFindCornerParticleMaterial(int4* codes, int4* cellpos, scalar4* pospres, scalar4* velrhop, scalar4* idmassidx, tensor2d* str, tensor2dful* contact_mag, scalar4* contact_var, 
	const int* cell_beg, const int* cell_end) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < par.npm) {
		int i = par.npb + index;
		// Information of particle i
		scalar3 pos_i = make_scalar3(pospres[i]);
		scalar4 velrhop_soil_i = velrhop[i];
		tensor2d str_over_rho2_i = str[i] / (velrhop_soil_i.w * velrhop_soil_i.w);
		scalar3 acc_s_i = make_scalar3(0.0f); // acceleration from stress 
		contact_var[i].x = 0.0;// reset integrate gker of external part
		contact_var[i].y = 999;//reset distance along ni
		scalar d_ij = 999;//reset distance_abs

		int3 celli = make_int3(cellpos[i]);

		for (int cz = -1; cz <= 1; cz++) {
			for (int cy = -1; cy <= 1; cy++) {
				for (int cx = -1; cx <= 1; cx++) {
					int3 cell_offset = make_int3(cx, cy, cz);
					int3 cellj = celli + cell_offset;
					if ((!(cellj >= make_int3(0))) || (!(cellj < par.cell_num)))continue;
					
					int cell_idx = cellj.x * par.cell_num.y * par.cell_num.z + cellj.y * par.cell_num.z + cellj.z;
					cuFindCornerParticleLoop(cell_beg[cell_idx], cell_end[cell_idx], pos_i, cell_offset, codes, cellpos, pospres, velrhop, idmassidx, str_over_rho2_i, acc_s_i, str, contact_mag, contact_var, d_ij, i, true);
					//int cell_idx_material = cell_idx + par.ncell_material;
					//cuFindCornerParticleLoop(cell_beg[cell_idx_material], cell_end[cell_idx_material], pos_i, cell_offset, codes, cellpos, pospres, velrhop, idmassidx, str_over_rho2_i, acc_s_i, str, contact_mag, contact_var, d_ij, i, false);
				}
			}
		}
	}
}

void LoquatSph::FindCornerParticle()
{
	if (par_host->npb == 0) return;

	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->npm + block.x - 1) / block.x, 1);
	if (par_host->npm != 0) {
		cuFindCornerParticleMaterial << <cuda_grid_size, block, 0 >> > (codes, cellpos, pospres, velrhop, idmassidx, str, contact_mag, contact_var, grid->GetCellBeg(), grid->GetCellEnd());
		getLastCudaError("cuFindCornerParticleMaterial failed.....     \n");
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*static __inline__ __device__ void cuIntegrateParticleMaterialLoop(int part_beg, int part_end, scalar3 pos_i, int3 cell_offset, int4* codes, int4* cellpos, scalar4* pospres,
	scalar4* velrhop, scalar4* idmassidx, const tensor2d str_over_rho2_i, scalar3& acc_s_i, tensor2d* str, tensor2dful* contact_mag, scalar4* contact_var, scalar& d_ij, const int i, bool bound) {
	for (int j = part_beg; j < part_end; j++) {

		scalar3 pos_j = make_scalar3(pospres[j]);
		scalar3 dr = pos_i - pos_j - cell_offset * (2.0 * par.h);
		scalar r = norm3df(dr.x, dr.y, dr.z);
		scalar q = r * par.i_h;

		if (q < 2.0) {
			scalar ker = GetKernel(par.kernel_normalization_par, q);
			scalar dwdr = GetGradKernel(par.kernel_gradient_normalization_par, q);
			scalar3 gker = dr * dwdr;

			//sigma integration of material for structure particles
			scalar rho_j = velrhop[j].w;
			tensor2d str_over_rho2_j = str[j] / (rho_j * rho_j);
			tensor2d str_term = str_over_rho2_i + str_over_rho2_j;
			acc_s_i = acc_s_i + DotProduct(str_term, gker) * idmassidx[j].y;

		}
	}
}

__global__ void cuIntegrateParticleMaterial(int4* codes, int4* cellpos, scalar4* pospres, scalar4* velrhop, scalar4* idmassidx, tensor2d* str, tensor2dful* contact_mag, scalar4* contact_var,
	const int* cell_beg, const int* cell_end) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < par.npb && codes[i].x >= 10) {
		// Information of particle i
		scalar3 pos_i = make_scalar3(pospres[i]);
		scalar4 velrhop_soil_i = velrhop[i];
		tensor2d str_over_rho2_i = str[i] / (velrhop_soil_i.w * velrhop_soil_i.w);
		scalar3 acc_s_i = make_scalar3(0.0f); // acceleration from stress 
		//contact_var[i].x = 0.0;// reset integrate gker of external part
		contact_var[i].y = 999;//reset distance along ni
		scalar d_ij = 999;//reset distance_abs

		int3 celli = make_int3(cellpos[i]);

		for (int cz = -1; cz <= 1; cz++) {
			for (int cy = -1; cy <= 1; cy++) {
				for (int cx = -1; cx <= 1; cx++) {
					int3 cell_offset = make_int3(cx, cy, cz);
					int3 cellj = celli + cell_offset;
					if ((!(cellj >= make_int3(0))) || (!(cellj < par.cell_num)))continue;

					int cell_idx = cellj.x * par.cell_num.y * par.cell_num.z + cellj.y * par.cell_num.z + cellj.z;
					//cuIntegrateParticleMaterialLoop(cell_beg[cell_idx], cell_end[cell_idx], pos_i, cell_offset, codes, cellpos, pospres, velrhop, idmassidx, str_over_rho2_i, acc_s_i, str, contact_mag, contact_var, d_ij, i, true);
					int cell_idx_material = cell_idx + par.ncell_material;
					cuIntegrateParticleMaterialLoop(cell_beg[cell_idx_material], cell_end[cell_idx_material], pos_i, cell_offset, codes, cellpos, pospres, velrhop, idmassidx, str_over_rho2_i, acc_s_i, str, contact_mag, contact_var, d_ij, i, false);
				}
			}
		}
		//sigma integration of material for structure particles
		contact_mag[i].xx = acc_s_i.x;
		contact_mag[i].xy = acc_s_i.y;
		contact_mag[i].xz = acc_s_i.z;
	}
}

void LoquatSph::IntegrateParticleMaterial()
{
	if (par_host->npb == 0) return;

	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->npb + block.x - 1) / block.x, 1);
	if (par_host->npm != 0) {
		cuIntegrateParticleMaterial << <cuda_grid_size, block, 0 >> > (codes, cellpos, pospres, velrhop, idmassidx, str, contact_mag, contact_var, grid->GetCellBeg(), grid->GetCellEnd());
		getLastCudaError("cuIntegrateParticleMaterial failed.....     \n");
	}
}*/