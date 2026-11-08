/*********************************************************************************
 * Copyright (c) 2024, Chong Peng, Chengwei Zhu. All rights reserved.

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

__inline__ __device__ void cuParticleInteractionMaterialLoopSolution(const int part_beg, const int part_end, const int3 cell_offset, const scalar3 pos_i,
	const scalar4 velrhop_soil_i, const tensor2d str_over_rho2_i, scalar4* pospres, scalar4* velrhop_s, scalar4* idmassidx,
	tensor2d* str, scalar& accden, scalar3& acc_s_i, scalar3& acc_av_i, scalar3& acc_as_i, tensor2dful& grad_vel, const bool bound, int4* codes, const int contact_shift_i, const scalar3 normal_con, tensor2dful* contact_mag){
		for(int j = part_beg; j < part_end; j++){
			scalar3 pos_j = make_scalar3(pospres[j]);
			scalar3 dr = pos_i - pos_j - cell_offset * (2.0 * par.h);
			scalar r = norm3df(dr.x, dr.y, dr.z);
			scalar q = r * par.i_h;
			//bool contact = (codes[j].x >= 10) && (bound);
		
			if (q < 2.0f) {
				// Information about particle i & j
				scalar ker = GetKernel(par.kernel_normalization_par, q);
				scalar dwdr = GetGradKernel(par.kernel_gradient_normalization_par, q);

				scalar rho_i = velrhop_soil_i.w;
				scalar3 vel_i = make_scalar3(velrhop_soil_i);
				scalar3 gker = dr * dwdr;		
				scalar rho_j = velrhop_s[j].w;
				scalar3 vel_j = make_scalar3(velrhop_s[j]);
				scalar mass_j = bound ? par.vol_def * rho_j : idmassidx[j].y;
					
				scalar3 dvel_ss = vel_i - vel_j;
				tensor2d str_over_rho2_j = str[j] / (rho_j * rho_j);
				
				/*if (codes[j].x >= 10) {//correct vel of contact particles  by extrapolating
					scalar3 dvel_ss_n = normal_con * DotProduct(normal_con, dvel_ss);//dv_n
					scalar3 dvel_ss_t = dvel_ss - dvel_ss_n;//dv_t
					if (contact_shift_i == 1) {//shift with contact or not
						//vel_j.x = contact_mag[j].zx;
						//vel_j.y = contact_mag[j].zy;
						//vel_j.z = contact_mag[j].zz;
						dvel_ss = dvel_ss_n;
					}
					else {
						//vel_j = vel_i;
					}
					//dvel_ss = make_scalar3(0);
				}
				scalar3 dvel_ss = vel_i - vel_j;*/

				if (codes[j].x >= 10 && contact_shift_i == 1) {//shift with contact or not
					dvel_ss = make_scalar3(0);
				}
				
				scalar vol_j = mass_j / rho_j;
				// Velocity gradient
				grad_vel = grad_vel - TensorProduct(dvel_ss, gker) * vol_j; 
				
				if (codes[j].x < 10) {//not contact particle


					// Density rate
					accden += rho_i * DotProduct(dvel_ss, gker) * vol_j;
				
					// Acceleration from stress
					tensor2d str_term = str_over_rho2_i + str_over_rho2_j;
					acc_s_i = acc_s_i + DotProduct(str_term, gker) * mass_j;			
		
					// Acceleration from artificial viscosity
					if(par.viscosity_type == 1){
						scalar artv = 0.0f;
						scalar visc = DotProduct(dvel_ss, dr);
						if(visc < 0.0f){
							scalar phi = par.h * visc / (r * r + par.eta);
							artv = par.artificial_viscosity_alpha_m * par.cs_m * phi / (0.5f * (rho_i + rho_j));
						}
						acc_av_i = acc_av_i + gker * (mass_j * artv);
					}
		
					// Acceleration from artificial stress
					if(par.artificial_pressure_coefficient_material != 0.f)	{
						scalar artp = trace(str_over_rho2_i);
						if(artp > 0.0f) artp *= par.artificial_pressure_coefficient_material;
						else			artp = 0.0f;
		
						scalar rj = trace(str_over_rho2_j);
						if(rj > 0.0f) rj *= par.artificial_pressure_coefficient_material ;
						else		  rj = 0.0f;
		
						artp += rj;
						if(artp > 0.0f){
							artp *= powf(ker / par.kernel_dr, par.kernel_zero / par.kernel_dr);
							acc_as_i = acc_as_i - gker * (mass_j * artp);				
						}
					}
				}
			}
		}
}

__inline__ __device__ void cuParticleInteractionMaterialLoop(const int3 celli, const scalar3 pos_i, const scalar4 velrhop_soil_i, 
	const tensor2d str_over_rho2_i, scalar4* pospres, scalar4* velrhop_s, scalar4* idmassidx, tensor2d* str, const int* cell_beg, 
	const int* cell_end, scalar& accden, scalar3& acc_s_i, scalar3& acc_av_i, scalar3& acc_as_i, tensor2dful& grad_vel, int4* codes, const int contact_shift_i, const scalar3 normal_con, tensor2dful* contact_mag) {
	for (int cz = -1; cz <= 1; cz++) {
		for (int cy = -1; cy <= 1; cy++) {
			for (int cx = -1; cx <= 1; cx++) {
				int3 cell_offset = make_int3(cx, cy, cz);
				int3 cellj = celli + cell_offset;
				if ((!(cellj >= make_int3(0))) || (!(cellj < par.cell_num)))continue;

				// Interaction with boundary particles
				int cell_idx = cellj.x * par.cell_num.y * par.cell_num.z + cellj.y * par.cell_num.z + cellj.z;
				int cell_idx_bound = cell_idx;
				cuParticleInteractionMaterialLoopSolution(cell_beg[cell_idx_bound], cell_end[cell_idx_bound], cell_offset, pos_i, velrhop_soil_i, str_over_rho2_i, pospres, velrhop_s, idmassidx, str,
					accden, acc_s_i, acc_av_i, acc_as_i, grad_vel, true, codes, contact_shift_i, normal_con, contact_mag);

				// Interaction with material particles
				int cell_idx_material = cell_idx + par.ncell_material;
				cuParticleInteractionMaterialLoopSolution(cell_beg[cell_idx_material], cell_end[cell_idx_material], cell_offset, pos_i, velrhop_soil_i, str_over_rho2_i, pospres, velrhop_s, idmassidx, str,
					accden, acc_s_i, acc_av_i, acc_as_i, grad_vel, false, codes, contact_shift_i, normal_con, contact_mag);
			}
		}
	}
}

__inline__ __device__ void cuParticleInteractionMaterialLoopSolutionContact(const int part_beg, const int part_end, const int3 cell_offset, const scalar3 pos_i,
	const scalar4 velrhop_soil_i, const tensor2d str_over_rho2_i, scalar4* pospres, int4* cellpos, scalar4* velrhop_s, scalar4* idmassidx,
	tensor2d* str, scalar& accden, scalar3& acc_st_i, tensor2dful& grad_vel, const scalar3 pos_abs_i, const int i, int& count, scalar3& fric, tensor2dful* contact_mag, scalar4* contact_var) {
	for (int j = part_beg; j < part_end; j++) {

		if (contact_var[j].y <= par.dr * 1.0001f) {//a little bit more scale for active state 
			scalar3 pos_abs_j;
			pos_abs_j.x = pospres[j].x + cellpos[j].x * (2.0f * par.h);
			pos_abs_j.y = pospres[j].y + cellpos[j].y * (2.0f * par.h);
			pos_abs_j.z = pospres[j].z + cellpos[j].z * (2.0f * par.h);

			//column search
			scalar3 d_column = make_scalar3(pos_abs_i.x - pos_abs_j.x, pos_abs_i.y - pos_abs_j.y, pos_abs_i.z - pos_abs_j.z);
			scalar r_column = abs(d_column.x * contact_mag[i].yz + d_column.y * contact_mag[i].yy + d_column.z * contact_mag[i].yx);//exchange norm x z

			bool in_same_line = (r_column <= par.dr / 2);
			
			if (in_same_line && contact_var[j].x != 0) {
				if (contact_var[i].x > contact_var[j].x) {
					contact_var[i].x = contact_var[j].x;//max distributing rate = 1
				}

				fric.x = contact_mag[j].xx * contact_var[i].x / contact_var[j].x;
				fric.y = contact_mag[j].xy * contact_var[i].x / contact_var[j].x;
				fric.z = contact_mag[j].xz * contact_var[i].x / contact_var[j].x;

				count++;
			}
		}
	}
}

__inline__ __device__ void cuParticleInteractionMaterialLoopContact(const int3 celli, const scalar3 pos_i, const scalar4 velrhop_soil_i,
	const tensor2d str_over_rho2_i, scalar4* pospres, scalar4* velrhop_s, scalar4* idmassidx, tensor2d* str, const int* cell_beg,
	const int* cell_end, scalar& accden, scalar3& acc_st_i, tensor2dful& grad_vel, int4* cellpos, int4* codes, tensor2dful* contact_mag, scalar4* contact_var, const int i) {
	
	scalar3 pos_abs_i = pos_i + celli * (2.0f * par.h); //Transition from relative position to absolute position
	scalar3 norm_bound = make_scalar3(contact_mag[i].yx, contact_mag[i].yy, contact_mag[i].yz);//norm of boundary
	scalar3 vel_bound = make_scalar3(contact_mag[i].zx, contact_mag[i].zy, contact_mag[i].zz);// (contact_mag[i].zx, contact_mag[i].zy, contact_mag[i].zz);//Vel of boundary
	scalar3 fric = make_scalar3(0.0f);//Friction

	scalar mass_rate = 2.0f / (2.0f + 0.0f);//deformable or not
	scalar3 delta_v = make_scalar3((vel_bound.x - velrhop_soil_i.x) * mass_rate, (vel_bound.y - velrhop_soil_i.y) * mass_rate, (vel_bound.z - velrhop_soil_i.z) * mass_rate);
	int count = 0;//number of surface contact particles in the same column
	
	if (contact_var[i].y <= par.h * 2 && DotProduct(delta_v, norm_bound) > 0) {
		scalar fric_coef = 0.58108f;//tan16.43 = 0.29489 tan16.57 = 0.29754 tan30 = 0.57735 tan30.16 = 0.58108
		if (contact_var[i].w == 11) fric_coef = 0.58108f;

		scalar3 delta_v_n = make_scalar3(norm_bound.x * DotProduct(delta_v, norm_bound), norm_bound.y * DotProduct(delta_v, norm_bound), norm_bound.z * DotProduct(delta_v, norm_bound));
		scalar3 fric_n = delta_v_n * idmassidx[i].y / par.dt;
		scalar3 delta_v_t = delta_v - delta_v_n;
		scalar3 fric_t = delta_v_t * idmassidx[i].y / par.dt;

		scalar absvalue_fric_n = EulerianNorm(fric_n);
		scalar absvalue_fric_t = EulerianNorm(fric_t);
		if (absvalue_fric_t > (fric_coef * absvalue_fric_n)) {
			fric_t = fric_t * (fric_coef * absvalue_fric_n / absvalue_fric_t);
			contact_var[i].z = 1;//shift with structure
		}
		else {
			contact_var[i].z = 0;
		}

		fric = (fric_t + fric_n);

		//distribute fric to every layers in 2h
		if (contact_var[i].y > par.dr * 1.0f) {//a little bit more scale for active state 
			//int count = 0;
			for (int cz = -1; cz <= 1; cz++) {
				for (int cy = -1; cy <= 1; cy++) {
					for (int cx = -1; cx <= 1; cx++) {
						int3 cell_offset = make_int3(cx, cy, cz);
						int3 cellj = celli + cell_offset;
						if ((!(cellj >= make_int3(0))) || (!(cellj < par.cell_num)))continue;

						// Interaction with material particles
						int cell_idx = cellj.x * par.cell_num.y * par.cell_num.z + cellj.y * par.cell_num.z + cellj.z;
						int cell_idx_material = cell_idx + par.ncell_material;
						cuParticleInteractionMaterialLoopSolutionContact(cell_beg[cell_idx_material], cell_end[cell_idx_material], cell_offset, pos_i, velrhop_soil_i, str_over_rho2_i, pospres, cellpos, velrhop_s, idmassidx, str,
							accden, acc_st_i, grad_vel, pos_abs_i, i, count, fric, contact_mag ,contact_var);
					}
				}
			}
			if (count == 0) {
				fric.x *= 0;
				fric.y *= 0;
				fric.z *= 0;
			}
		}
		//Output true fric
		contact_mag[i].xx = fric.x;
		contact_mag[i].xy = fric.y;
		contact_mag[i].xz = fric.z;

		acc_st_i = fric * (1 / idmassidx[i].y);
	}

	//idmassidx[i].z = count;
}

__inline__ __device__ void cuParticleInteractionMaterialLoopPeriod(const int3 celli, const scalar3 pos_i, const scalar4 velrhop_soil_i,
	const tensor2d str_over_rho2_i, scalar4* pospres, scalar4* velrhop_s, scalar4* idmassidx, tensor2d* str, const int* cell_beg,
	const int* cell_end, scalar& accden, scalar3& acc_s_i, scalar3& acc_av_i, scalar3& acc_as_i, tensor2dful& grad_vel, int4* codes, const int contact_shift_i, const scalar3 normal_con, tensor2dful* contact_mag) {
	scalar3 pos_abs_i = pos_i + celli * (2.0 * par.h); //Transition from relative position to absolute position
	bool period_left = AbsDist2Plane(pos_abs_i, par.planeleft) < 2.0 * par.h;
	bool period_right = AbsDist2Plane(pos_abs_i, par.planeright) < 2.0 * par.h;
	if (period_left || period_right) {
		scalar shift_length = par.planeleft.w + par.planeright.w;
		scalar3 shift_vector = period_left ? make_scalar3(par.planeleft) * shift_length : make_scalar3(par.planeright) * shift_length;
		scalar3 pos_abs_pb_i = pos_abs_i + shift_vector;
		int3 cell_pb_i = make_int3(pos_abs_pb_i * (0.5 * par.i_h));
		scalar3 pos_pb_i = pos_abs_pb_i - cell_pb_i * (2.0 * par.h);
		cuParticleInteractionMaterialLoop(cell_pb_i, pos_pb_i, velrhop_soil_i, str_over_rho2_i, pospres, velrhop_s, idmassidx, str, cell_beg, cell_end,
			accden, acc_s_i, acc_av_i, acc_as_i, grad_vel, codes, contact_shift_i, normal_con, contact_mag);
	}
}

__global__ void cuParticleInteractionMaterial(int4* codes, tensor2dful* contact_mag, scalar4* contact_var, int4* cellpos, scalar4* pospres, scalar4* velrhop, scalar4* idmassidx, tensor2d* str, tensor2d* stn,
	scalar4* accad, scalar4* spin, tensor2d* acestn, const int* cell_beg, const int* cell_end){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < par.npm){
		int i = par.npb + index;
		int3 celli = make_int3(cellpos[i]);
		scalar3 pos_i = make_scalar3(pospres[i]);
		scalar4 velrhop_soil_i = velrhop[i];
		tensor2d str_over_rho2_i = str[i] / (velrhop_soil_i.w * velrhop_soil_i.w);

		scalar accden = 0.f; // density rate
		scalar3 acc_s_i = make_scalar3(0.0f); // acceleration from stress 
		scalar3 acc_av_i = make_scalar3(0.0f); // acceleration from artificial viscosity
		scalar3 acc_as_i = make_scalar3(0.0f); // acceleration from artificial stress
		scalar3 acc_st_i = make_scalar3(0.0f); // acceleration from structure contact
		tensor2dful grad_vel = make_tensor2dful(0.0f);
		int contact_shift_i = contact_var[i].z;//shift with contact or not
		scalar3 normal_con = make_scalar3(-contact_mag[i].yx, -contact_mag[i].yy, -contact_mag[i].yz); // normal vector unit of contact

		cuParticleInteractionMaterialLoop(celli, pos_i, velrhop_soil_i, str_over_rho2_i, pospres, velrhop, idmassidx, str, cell_beg, cell_end,
			accden, acc_s_i, acc_av_i, acc_as_i, grad_vel, codes, contact_shift_i, normal_con, contact_mag);
		if(par.periodbound)cuParticleInteractionMaterialLoopPeriod(celli, pos_i, velrhop_soil_i, str_over_rho2_i, pospres, velrhop, idmassidx, str, cell_beg, cell_end,
			accden, acc_s_i, acc_av_i, acc_as_i, grad_vel, codes, contact_shift_i, normal_con, contact_mag);

		//update vel without contact
		scalar3 acc_damping = make_scalar3(velrhop_soil_i) * par.damping;
		accad[i] = make_scalar4(acc_s_i + acc_av_i + acc_as_i - acc_damping + par.gravity, accden);
		velrhop_soil_i = velrhop_soil_i + accad[i] * par.dt;

		cuParticleInteractionMaterialLoopContact(celli, pos_i, velrhop_soil_i, str_over_rho2_i, pospres, velrhop, idmassidx, str, cell_beg, cell_end,
			accden, acc_st_i, grad_vel, cellpos, codes, contact_mag, contact_var, i);

		
		accad[i] = make_scalar4(acc_s_i + acc_av_i + acc_as_i + acc_st_i - acc_damping + par.gravity, accden);
		acestn[i] = SymTensor(grad_vel);					  
		spin[i] = make_scalar4(SkewTensor(grad_vel), 0.f);
	}
}

void ParticleInteractionMaterial(Parameters* par_host, int4* codes, tensor2dful* contact_mag, scalar4* contact_var, int4* cellpos, scalar4* pospres,	scalar4* velrhop, scalar4* idmassidx,	tensor2d* str, tensor2d* stn,
	scalar4* accad, scalar4* spin, tensor2d* acestn, const int* cell_beg, const int* cell_end){
	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->npm + block.x - 1) / block.x, 1);
	cuParticleInteractionMaterial <<<cuda_grid_size, block, 0>>> (codes, contact_mag, contact_var, cellpos, pospres, velrhop, idmassidx, str, stn, accad, spin, acestn, cell_beg, cell_end);
	getLastCudaError("cuParticleInteractionMaterial failed...     \n");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LoquatSph::ParticleInteraction(){
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));
	if(par_host->npm != 0) ParticleInteractionMaterial(par_host, codes, contact_mag, contact_var, cellpos, pospres, velrhop, idmassidx, str, stn, accad, spin, acestn, grid->GetCellBeg(), grid->GetCellEnd());
}
