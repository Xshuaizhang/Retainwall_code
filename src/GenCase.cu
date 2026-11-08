
#include "types.h"
#include "functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

void GetMaterial_DP_EPP(scalar E, scalar V, scalar phi, scalar psi, scalar cohesion, scalar rho_min, scalar rho_max, material DP_EPP_model) {
	scalar tanphi = tan(phi), tanpsi = tan(psi);
	DP_EPP_model[0] = E / (3.0f * (1.0f - 2.0f * V));
	DP_EPP_model[1] = E / (2.0f * (1.0f + V));
	DP_EPP_model[2] = 3.0 * tanphi / sqrt(9.0 + 12.0 * tanphi * tanphi);
	DP_EPP_model[3] = 3.0 * tanpsi / sqrt(9.0 + 12.0 * tanpsi * tanpsi);
	DP_EPP_model[4] = 3.0 * cohesion / sqrt(9.0 + 12.0 * tanphi * tanphi);
	DP_EPP_model[MATERIAL_SIZE_MAX-2] = rho_min;
	DP_EPP_model[MATERIAL_SIZE_MAX-1] = rho_max;
}

void GetMaterial_MuI_DP_EPP(scalar E, scalar V, scalar phi1, scalar phi2, scalar I0, scalar rho_g, scalar d_grain, scalar lambda, scalar rho_min, scalar rho_max, material MuI_DP_EPP_model) {
	scalar tanphi1 = tan(phi1 * PI_D / 180.0);
	scalar tanphi2 = tan(phi2 * PI_D / 180.0);
	MuI_DP_EPP_model[0] = E / (3.0 * (1.0 - 2.0 * V));
	MuI_DP_EPP_model[1] = E / (2.0 * (1.0 + V));
	MuI_DP_EPP_model[2] = tanphi1;
	MuI_DP_EPP_model[3] = tanphi2;
	MuI_DP_EPP_model[4] = I0;    
	MuI_DP_EPP_model[5] = rho_g;    
	MuI_DP_EPP_model[6] = d_grain;    
	MuI_DP_EPP_model[7] = lambda;  
	MuI_DP_EPP_model[MATERIAL_SIZE_MAX - 2] = rho_min;
	MuI_DP_EPP_model[MATERIAL_SIZE_MAX - 1] = rho_max;
}

void GetMaterial_HypoBasic(scalar Ei, scalar Vi, scalar phi_c, scalar cohesion, scalar p_ref, scalar rho_min, scalar rho_max, material HypoBasic_model) {
	HypoBasic_model[0] = Ei;
	HypoBasic_model[1] = Vi;
	HypoBasic_model[2] = phi_c;
	HypoBasic_model[3] = cohesion;
	HypoBasic_model[4] = p_ref;
	HypoBasic_model[MATERIAL_SIZE_MAX - 2] = rho_min;
	HypoBasic_model[MATERIAL_SIZE_MAX - 1] = rho_max;
}

void GetMaterial_HypoCS(scalar Ei, scalar Vi, scalar phi_c, scalar cohesion, scalar p_ref, scalar void_co, scalar lambda, scalar xi, scalar alpha, scalar void_min, scalar void_max, scalar void_0, scalar rho0, scalar rho_min, scalar rho_max, material HypoCS_model) {
	HypoCS_model[0] = Ei;
	HypoCS_model[1] = Vi;
	HypoCS_model[2] = phi_c;
	HypoCS_model[3] = cohesion;
	HypoCS_model[4] = p_ref;
	HypoCS_model[5] = void_co;
	HypoCS_model[6] = lambda;
	HypoCS_model[7] = xi;
	HypoCS_model[8] = alpha;
	HypoCS_model[9] = void_min;
	HypoCS_model[10] = void_max;
	HypoCS_model[11] = void_0;
	HypoCS_model[12] = rho0;
	HypoCS_model[MATERIAL_SIZE_MAX - 2] = rho_min;
	HypoCS_model[MATERIAL_SIZE_MAX - 1] = rho_max;
}

void SaveDatFile(std::string& casename,
	Parameters* par_host,
	std::vector<int4>& codes,
	std::vector<scalar4>& pospres,
	std::vector<scalar4>& velrhop,
	std::vector<scalar4>& idmassidx,
	std::vector<tensor2d>& stress,
	std::vector<tensor2d>& strain,
	std::vector<scalar4>& normal_m)
{
	std::string filename = casename;
	filename.append(".dat");

	std::ofstream ofs(filename.c_str(), std::ofstream::out);
	if (ofs)
	{
		for (int i = 0; i < par_host->npb; i++) // boundary
		{
			// x y z p vx vy vz rho id mass ep phi str stn normal
			ofs << codes[i].x << " "<< codes[i].y << " " << codes[i].z << " " << codes[i].w << " "
				<< pospres[i].x << " " << pospres[i].y << " " << pospres[i].z << " " << pospres[i].w << " "
				<< velrhop[i].x << " " << velrhop[i].y << " " << velrhop[i].z << " " << velrhop[i].w << " "
				<< idmassidx[i].x << " " << idmassidx[i].y << " " << idmassidx[i].w << " "
				<< stress[i].xx << " " << stress[i].yy << " " << stress[i].zz << " " << stress[i].xy << " " << stress[i].xz << " " << stress[i].yz << " "
				<< strain[i].xx << " " << strain[i].yy << " " << strain[i].zz << " " << strain[i].xy << " " << strain[i].xz << " " << strain[i].yz << " "
				<< normal_m[i].x << " " << normal_m[i].y << " " << normal_m[i].z << " " << normal_m[i].w << std::endl;
		}

		for (int i = par_host->npb; i < par_host->npb + par_host->npm; i++) // material
		{
			// x y z p vx vy vz rho id mass ep phi str stn
			ofs << codes[i].x << " " << codes[i].y << " " << codes[i].z << " " << codes[i].w << " "
				<< pospres[i].x << " " << pospres[i].y << " " << pospres[i].z << " " << pospres[i].w << " "
				<< velrhop[i].x << " " << velrhop[i].y << " " << velrhop[i].z << " " << velrhop[i].w << " "
				<< idmassidx[i].x << " " << idmassidx[i].y << " " << idmassidx[i].w << " "
				<< stress[i].xx << " " << stress[i].yy << " " << stress[i].zz << " " << stress[i].xy << " " << stress[i].xz << " " << stress[i].yz << " "
				<< strain[i].xx << " " << strain[i].yy << " " << strain[i].zz << " " << strain[i].xy << " " << strain[i].xz << " " << strain[i].yz << std::endl;
		}

		ofs.close();
	}
	else
	{
		std::cout << "Error! Cannot write in " << filename << " !" << std::endl;
		exit(EXIT_FAILURE);
	}
}


void SaveXmlFile(std::string casename, Parameters* par_host)
{
	std::string filename = casename;
	filename.append(".xml");

	std::ofstream ofs(filename.c_str(), std::ofstream::out);
	if (ofs)
	{
		ofs << "<?xml version=\"1.0\" encoding=\"utf-8\" ?>" << std::endl;
		ofs << "<parameters>" << std::endl;

		// Configuration
		ofs << "  <dimension-of-problem type=\"dimension\">" << par_host->dim << "</dimension-of-problem>" << std::endl;
		ofs << "  <number-of-motion type=\"int\">" << par_host->n_motion << "</number-of-motion>" << std::endl;
		ofs << "  <period-bound type=\"bool\">" << par_host->periodbound << "</period-bound>" << std::endl;
		if (par_host->periodbound) {
			ofs << "  <planeleft-A type=\"scalar\">" << par_host->planeleft.x << "</planeleft-A>" << std::endl;
			ofs << "  <planeleft-B type=\"scalar\">" << par_host->planeleft.y << "</planeleft-B>" << std::endl;
			ofs << "  <planeleft-C type=\"scalar\">" << par_host->planeleft.z << "</planeleft-C>" << std::endl;
			ofs << "  <planeleft-D type=\"scalar\">" << par_host->planeleft.w << "</planeleft-D>" << std::endl;
			ofs << "  <planeright-A type=\"scalar\">" << par_host->planeright.x << "</planeright-A>" << std::endl;
			ofs << "  <planeright-B type=\"scalar\">" << par_host->planeright.y << "</planeright-B>" << std::endl;
			ofs << "  <planeright-C type=\"scalar\">" << par_host->planeright.z << "</planeright-C>" << std::endl;
			ofs << "  <planeright-D type=\"scalar\">" << par_host->planeright.w << "</planeright-D>" << std::endl;
		}
		ofs << "  <number-of-particles type=\"int\">" << par_host->np << "</number-of-particles>" << std::endl;
		ofs << "  <number-of-boundary-particles type=\"int\">" << par_host->npb << "</number-of-boundary-particles>" << std::endl;
		ofs << "  <number-of-material-particles type=\"int\">" << par_host->npm << "</number-of-material-particles>" << std::endl;

		ofs << "  <domain-size-x type=\"scalar\">" << par_host->domain_size.x << "</domain-size-x>" << std::endl;
		ofs << "  <domain-size-y type=\"scalar\">" << par_host->domain_size.y << "</domain-size-y>" << std::endl;
		ofs << "  <domain-size-z type=\"scalar\">" << par_host->domain_size.z << "</domain-size-z>" << std::endl;
		ofs << "  <domain-min-x type=\"scalar\">" << par_host->domain_min.x << "</domain-min-x>" << std::endl;
		ofs << "  <domain-min-y type=\"scalar\">" << par_host->domain_min.y << "</domain-min-y>" << std::endl;
		ofs << "  <domain-min-z type=\"scalar\">" << par_host->domain_min.z << "</domain-min-z>" << std::endl;

		ofs << "  <particle-size type=\"scalar\">" << par_host->dr << "</particle-size>" << std::endl;
		ofs << "  <h-over-dr type=\"scalar\">" << par_host->hdr << "</h-over-dr>" << std::endl;

		ofs << "  <acceleration-x type=\"scalar\">" << par_host->gravity.x << "</acceleration-x>" << std::endl;
		ofs << "  <acceleration-y type=\"scalar\">" << par_host->gravity.y << "</acceleration-y>" << std::endl;
		ofs << "  <acceleration-z type=\"scalar\">" << par_host->gravity.z << "</acceleration-z>" << std::endl;

		// Material parameters
		ofs << "  <reference-density-material type=\"scalar\">" << par_host->rho0_m << "</reference-density-material>" << std::endl;
		ofs << "  <damping type=\"scalar\">" << par_host->damping << "</damping>" << std::endl;
		ofs << "  <number-of-model type=\"int\">" << par_host->n_model << "</number-of-model>" << std::endl;
		for (int i = 0; i < par_host->n_model; i++) {
			for (int j = 0; j < MATERIAL_SIZE_MAX; j++) {
				std::stringstream model_id;
				model_id << "model-" << i << "-par-" << j;
				ofs << "  <" << model_id.str() << " type=\"scalar\">" << par_host->model_par[i][j] << "</" << model_id.str() << ">" << std::endl;
			}
		}

		// Numerical stabilization techniques
		ofs << "  <stress-shepard-filter type=\"int\">" << par_host->stress_shepard_filter << "</stress-shepard-filter>" << std::endl;
		ofs << "  <dynamic-time-step type=\"int\">" << par_host->dynamic_time_step << "</dynamic-time-step>" << std::endl;
		ofs << "  <integration-method type=\"int\">" << par_host->integration_method << "</integration-method>" << std::endl;
		ofs << "  <viscosity-type type=\"int\">" << par_host->viscosity_type << "</viscosity-type>" << std::endl;
		if (par_host->viscosity_type == 1)
		{
			ofs << "  <artificial-viscosity-alpha-material type=\"scalar\">" << par_host->artificial_viscosity_alpha_m << "</artificial-viscosity-alpha-material>" << std::endl;
		}
		ofs << "  <speed-of-sound-material type=\"scalar\">" << par_host->cs_m << "</speed-of-sound-material>" << std::endl;
		ofs << "  <cfl-number type=\"scalar\">" << par_host->cfl << "</cfl-number>" << std::endl;
		ofs << "  <artificial-pressure-coefficient-material type=\"scalar\">" << par_host->artificial_pressure_coefficient_material << "</artificial-pressure-coefficient-material>" << std::endl;

		// Simulation output
		ofs << "  <time-max type=\"scalar\">" << par_host->time_max << "</time-max>" << std::endl;
		ofs << "  <time type=\"scalar\">" << static_cast<scalar>(par_host->time) << "</time>" << std::endl;
		ofs << "  <time-step type=\"scalar\">" << par_host->dt << "</time-step>" << std::endl;
		ofs << "  <output-frequency type=\"scalar\">" << par_host->output_frequency << "</output-frequency>" << std::endl;
		ofs << "  <output-number type=\"int\">" << par_host->output_number << "</output-number>" << std::endl;
		ofs << "  <output-format type=\"int\">" << par_host->output_format << "</output-format>" << std::endl;

		ofs << "</parameters>" << std::endl;

		ofs.close();
	}
	else
	{
		std::cout << "Error! Cannot write file " << filename << " !" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void SaveMotionFile(std::string casename, Parameters* par_host, std::vector<MotionTable>& mt) {
	std::cout << "Number of moving rigid body: " << par_host->n_motion << std::endl;
	std::string motion_filename = casename;
	motion_filename.append(".motion");
	std::ofstream motion_ofs(motion_filename.c_str(), std::ofstream::out);
	if (motion_ofs) {
		for (int i = 0; i < par_host->n_motion; i++) {
			motion_ofs << mt[i].is_translation << " " << mt[i].is_rotation << std::endl;
			if (mt[i].is_translation) {
				motion_ofs << mt[i].table_size_trans << std::endl;
				for (int j = 0; j < mt[i].table_size_trans; j++) motion_ofs << mt[i].trans_time[j] << " " << mt[i].trans_vel[j].x << " " << mt[i].trans_vel[j].y << " " << mt[i].trans_vel[j].z << " " << mt[i].trans_acc[j].x << " " << mt[i].trans_acc[j].y << " " << mt[i].trans_acc[j].z << std::endl;
			}
			if (mt[i].is_rotation) {
				motion_ofs << mt[i].pattern << " "
					<< mt[i].pos_rp.x << " " << mt[i].pos_rp.y << " " << mt[i].pos_rp.z << " "
					<< mt[i].rotat_vel.x << " " << mt[i].rotat_vel.y << " " << mt[i].rotat_vel.z << " " << mt[i].table_size_rotat << std::endl;
				for (int j = 0; j < mt[i].table_size_rotat; j++) motion_ofs << mt[i].rotat_time[j] << " " << mt[i].rotat_acc[j].x << " " << mt[i].rotat_acc[j].y << " " << mt[i].rotat_acc[j].z << std::endl;
			}
		}
	}
	motion_ofs.close();
}

void GetParticles(Parameters* par_host,
	scalar4*& pospres)
{
	int npx = int(par_host->domain_size.x / par_host->dr);
	int npy = int(par_host->domain_size.y / par_host->dr);
	int npz = int(par_host->domain_size.z / par_host->dr);

	if (npx == 0) npx = 1;
	if (npy == 0) npy = 1;

	par_host->np = npx * npy * npz;

	cudaHostAlloc((void**)&pospres, par_host->np * sizeof(scalar4), cudaHostAllocDefault);

	int idx = 0;
	scalar half_dr = 0.5f * par_host->dr;
	for (int i = 0; i < npx; i++)
	{
		for (int j = 0; j < npy; j++)
		{
			for (int k = 0; k < npz; k++)
			{
				pospres[idx] = make_scalar4(half_dr + par_host->dr * static_cast<scalar>(i),
					half_dr + par_host->dr * static_cast<scalar>(j),
					half_dr + par_host->dr * static_cast<scalar>(k),
					0.0f);
				idx++;
			}
		}
	}
}


int main(int argc, char* argv[])
{
	int case_num;
	scalar dr;
	scalar hdr;
	scalar vol;
	scalar bound_thickness;
	dimension dim=TwoD;

	if (argc < 4)
	{
		std::cout << "Error! Please specify built-in case number, particle size and hdr!" << std::endl;
		exit(EXIT_FAILURE);
	}
	else
	{
		case_num = atoi(argv[1]);
		dr = atof(argv[2]);
		hdr = (scalar)(atof(argv[3]));
		if (argc == 5)dim = (dimension)atoi(argv[4]);
		vol = dim==OneD ? dr : (dim==TwoD ? dr * dr : dr * dr * dr);
		bound_thickness = ceil(2.f * hdr) * dr;
	}

	std::cout << "Case number = " << case_num << ", dr = " << dr << ", hdr = " << hdr << std::endl;

	Parameters* par_host = new Parameters;

	// Default values
	par_host->dim = dim;
	par_host->periodbound = false;
	par_host->n_motion = 0; 
	par_host->dr = dr;
	par_host->hdr = hdr;
	par_host->h = par_host->dr * par_host->hdr;

	par_host->domain_min.x = 0.0f;
	par_host->domain_min.y = 0.0f;
	par_host->domain_min.z = 0.0f;

	par_host->gravity = make_scalar3(0.0f, 0.0f, -9.81f); // Gravity

	// Material parameters
	scalar rho0_m = 1500.0f;
	par_host->rho0_m = rho0_m; // Particle density of soil
	par_host->damping = 0.f;    // Damping	
	for (int i = 0; i < MODEL_NUMBER_MAX; i++)	std::memset(par_host->model_par[i], 0, sizeof(scalar) * MATERIAL_SIZE_MAX);
	material DP_EPP_model_0 = { 0 }; // Default DP_EPP material. You can change it here, or paste the code into the specific case for modification 
	{
		scalar E = 1.5e6f, V = 0.2f, phi = 25.0f * PI_D / 180.0, psi = 0.0f * PI_D / 180.0, cohesion = 0.0f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
		GetMaterial_DP_EPP(E, V, phi, psi, cohesion, rho_min, rho_max, DP_EPP_model_0);
	}
	par_host->n_model = 1;
	model_mat material1 = DP_EPP;
	std::copy(std::begin(DP_EPP_model_0), std::end(DP_EPP_model_0), par_host->model_par[0]);

	// Numerical stabilization techniques
	par_host->stress_shepard_filter = 0;
	par_host->dynamic_time_step = 1;
	par_host->integration_method = 1; // Predictor-Corrector
	par_host->viscosity_type = 1; // Artfificial viscosity
	par_host->artificial_viscosity_alpha_m = 0.25f;
	par_host->cs_m = 30.0f;
	par_host->cfl = 0.05f;

	par_host->artificial_pressure_coefficient_material = 0.0f;

	// Simulation output
	par_host->time = 0.0f;
	par_host->dt = 1.0e-6f;
	par_host->output_frequency = 0.02f;
	par_host->output_number = 0;
	par_host->output_format = 2; // Vtu

	std::string casename;
	std::vector<int4> vec_codes;
	std::vector<scalar4> vec_pospres, vec_velrhop, vec_idmassidx, vec_normal_m;
	std::vector<tensor2d> vec_stress, vec_strain;
	std::vector<MotionTable> vec_mt;

	std::cout << "Default parameters set" << std::endl;

	int npb = 0;
	int npm = 0;

	tensor2d zeros = make_tensor2d(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

	switch (case_num)
	{
	case 1: { // sand column collapse; this case is used to valiate the default constitutive models			
		if (dim==TwoD) {
			casename.append("sand_column_collapse_2D");
			std::cout << "Case: sand_column_collapse_2D" << std::endl;
		}
		else if (dim == ThreeD) {
			casename.append("sand_column_collapse_3D");
			std::cout << "Case: sand_column_collapse_3D" << std::endl;
		}

		// Initial Configuration
		scalar box_length = 0.5f;
		scalar box_height = 0.15f;
		scalar box_width = 0.05f;
		scalar column_length = 0.2f;
		scalar column_height = 0.1f;

		// Domain size
		par_host->domain_size.x = box_length + 2.0f * bound_thickness;
		par_host->domain_size.y = dim == TwoD ? 0.0f : box_width + 2.0f * bound_thickness;
		par_host->domain_size.z = box_height + 2.0f * bound_thickness;

		// Material parameters
		par_host->rho0_m = 1500.0f;
		scalar rho0_m = par_host->rho0_m;
		material DP_EPP_model_0 = { 0 }; // Default DP_EPP material. 
		{
			scalar E = 1.5e6f, V = 0.2f, phi = 25.0f * PI_D / 180.0, psi = 0.0f * PI_D / 180.0, cohesion = 0.0f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
			GetMaterial_DP_EPP(E, V, phi, psi, cohesion, rho_min, rho_max, DP_EPP_model_0);
		}

		material MuI_DP_EPP_model_0 = { 0 }; // Default MuI_DP_EPP material. 
		{
			scalar E = 1.5e6f, V = 0.2f, phi1 = 20.0, phi2 = 30.0, I0 = 0.003, rho_g = 2500.0, d_grain = 0.01, lambda = 0.001, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
			GetMaterial_MuI_DP_EPP(E, V, phi1, phi2, I0, rho_g, d_grain, lambda, rho_min, rho_max, MuI_DP_EPP_model_0);
		}

		material HypoBasic_model_0 = { 0 };  // Default HypoBasic material. 
		{
			scalar Ei = 1.5e8f, Vi = 0.2f, phi_c = 30.f * PI_D / 180.0, cohesion = 0.f, p_ref = 1.e5f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
			GetMaterial_HypoBasic(Ei, Vi, phi_c, cohesion, p_ref, rho_min, rho_max, HypoBasic_model_0);
		}

		material HypoCS_model_0 = { 0 };   // Default HypoCS material. 
		{
			scalar Ei = 1.5e8f, Vi = 0.2f, phi_c = 30.f * PI_D / 180.0, cohesion = 0.f, p_ref = 1.e5f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;                           // Basic material parameters
			scalar  void_co = 0.95, lambda = 0.122, xi = 0.7, alpha = 0.5, void_min = 0.6, void_max = 1.2, void_0 = 0.9, rho0 = par_host->rho0_m;                  // Material parameters related to critical state
			GetMaterial_HypoCS(Ei, Vi, phi_c, cohesion, p_ref, void_co, lambda, xi, alpha, void_min, void_max, void_0, rho0, rho_min, rho_max, HypoCS_model_0);
		}

		par_host->n_model = 2;
		model_mat material1 = HypoBasic;
		model_mat material2 = DP_EPP;
		std::copy(std::begin(HypoBasic_model_0), std::end(HypoBasic_model_0), par_host->model_par[0]);  // Herein, DP_EPP_model_0 can be replaced as some other model
		std::copy(std::begin(DP_EPP_model_0), std::end(DP_EPP_model_0), par_host->model_par[1]);  // Herein, DP_EPP_model_0 can be replaced as some other model
		
		// Numerical stabilization techniques
		par_host->cs_m = 30.0f;
		par_host->cfl = 0.1f;
		par_host->viscosity_type = 1;
		par_host->artificial_viscosity_alpha_m = 0.25f;
		par_host->stress_shepard_filter = 50;
		par_host->artificial_pressure_coefficient_material = 0.2f;

		// Simulation output 
		par_host->time_max = 1.0f;
		par_host->time = 0.0f;
		par_host->dt = 1.0e-6f;
		par_host->output_frequency = par_host->time_max / 100.f;

		scalar4* pospres = NULL;

		std::cout << "Start filling domain with lattice" << std::endl;
		GetParticles(par_host, pospres);
		std::cout << "Lattice completed, in total " << par_host->np << " particles were generated" << std::endl;
		std::cout << par_host->np << " lattice particles are generated" << std::endl;
		std::cout << "Start selecting particles" << std::endl;

		// Boundary
		for (int i = 0; i < par_host->np; i++)
		{
			scalar4 pospres_i = pospres[i];		
			if (dim == TwoD) pospres_i.y = 0.0f;

			bool bound_left = dim == OneD ? false : pospres_i.x <= bound_thickness;
			bool bound_right = dim == OneD ? false : pospres_i.x >= par_host->domain_size.x - bound_thickness;
			bool bound_bottom = pospres_i.z <= bound_thickness;
			bool bound_front = dim == TwoD ? false : pospres_i.y <= bound_thickness;
			bool bound_back = dim == TwoD ? false : pospres_i.y >= par_host->domain_size.y - bound_thickness;

			if (bound_left || bound_right || bound_bottom || bound_front || bound_back)
			{
				vec_pospres.push_back(pospres_i);
				vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, 0.0f));
				vec_idmassidx.push_back(make_scalar4(0.0f, 0.0, 0.0f, 0.0f));
				vec_stress.push_back(zeros);
				vec_strain.push_back(zeros);

				int blockID;
				status block_status = Fix;
				scalar4 normal_i;				
				if (bound_bottom) {					
					blockID = 0; 				
					normal_i = make_scalar4(0.f, 0.f, 1.0f, -1.0f);
				}
				else if (bound_front) {
					blockID = 3; 
					normal_i = make_scalar4(0.f, 1.f, 0.0f, 1.0f);
				}
				else if (bound_back) {
					blockID = 4; 
					normal_i = make_scalar4(0.f, -1.f, 0.0f, 1.0f);
				}
				else if (bound_left) {
					blockID = 1;
					normal_i = make_scalar4(1.f, 0.f, 0.0f, -1.0f);
				}
				else {
					blockID = 2;
					normal_i = make_scalar4(-1.f, 0.f, 0.0f, 1.0f);
				}
				vec_codes.push_back(make_int4(blockID, block_status, INT_MIN, INT_MIN));
				vec_normal_m.push_back(normal_i);
				npb++;
			}
		}

		// Material
		for (int i = 0; i < par_host->np; i++)
		{
			scalar4 pospres_i = pospres[i];
			if (dim == TwoD) pospres_i.y = 0.0f;

			bool column_top = pospres_i.x >= bound_thickness && pospres_i.x <= bound_thickness + column_length &&
				(dim == TwoD ? true : (pospres_i.y >= bound_thickness && pospres_i.y <= par_host->domain_size.y - bound_thickness)) &&
				pospres_i.z >= bound_thickness && pospres_i.z <= bound_thickness + column_height && pospres_i.z >= bound_thickness + column_height * 0.5;
			bool column_bottom = pospres_i.x >= bound_thickness && pospres_i.x <= bound_thickness + column_length &&
				(dim == TwoD ? true : (pospres_i.y >= bound_thickness && pospres_i.y <= par_host->domain_size.y - bound_thickness)) &&
				pospres_i.z >= bound_thickness && pospres_i.z <= bound_thickness + column_height * 0.5;

			if (column_top|| column_bottom) {
				scalar mass_m = rho0_m * vol;				
				vec_pospres.push_back(pospres_i);
				vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, rho0_m));
				vec_idmassidx.push_back(make_scalar4(100.0f, mass_m, 0.0f, 100.0f));
				vec_stress.push_back(zeros);
				vec_strain.push_back(zeros);
				if(column_bottom)vec_codes.push_back(make_int4(6, NonFix, material1, 0));
				else vec_codes.push_back(make_int4(6, NonFix, material2, 1));
				npm++;
			}
		}

		cudaFreeHost(pospres);
		break;
	}

	case 2: { // soil stress	
		if (dim==OneD) {
			casename.append("soil_stress_1D");
			std::cout << "Case: soil_stress_1D" << std::endl;
		}
		else if (dim==TwoD) {
			casename.append("soil_stress_2D");
			std::cout << "Case: soil_stress_2D" << std::endl;
		}
		else if (dim==ThreeD) {
			casename.append("soil_stress_3D");
			std::cout << "Case: soil_stress_3D" << std::endl;
		}

		// Initial Configuration
		scalar box_length = 0.5f;
		scalar box_height = 1.0f;
		scalar box_width = 0.5f;
		scalar soil_height = 1.0f;

		// Domain size
		par_host->domain_size.x = dim==OneD ? 0.f : box_length + 2.0f * bound_thickness;
		par_host->domain_size.y = (dim==OneD || dim==TwoD) ? 0.f : box_width + 2.0f * bound_thickness;
		par_host->domain_size.z = box_height + 2.0f * bound_thickness;

		// Material parameters
		par_host->rho0_m = 1500.0f;
		scalar rho0_m = par_host->rho0_m;	
		material DP_EPP_model_0 = { 0 }; 
		{
			scalar E = 1.5e6f, V = 0.2f, phi = 25.0f * PI_D / 180.0, psi = 0.0f * PI_D / 180.0, cohesion = 0.0f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
			GetMaterial_DP_EPP(E, V, phi, psi, cohesion, rho_min, rho_max, DP_EPP_model_0);
			par_host->damping = 0.02 * sqrt(E / rho0_m) / par_host->h;
		}

		material HypoBasic_model_0 = { 0 };  
		{
			scalar Ei = 1.5e8f, Vi = 0.2f, phi_c = 30.f * PI_D / 180.0, cohesion = 0.f, p_ref = 1.e5f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
			GetMaterial_HypoBasic(Ei, Vi, phi_c, cohesion, p_ref, rho_min, rho_max, HypoBasic_model_0);
			par_host->damping = 0.02 * sqrt(Ei / rho0_m) / par_host->h;
		}

		par_host->n_model = 1;
		model_mat material1 = DP_EPP;
		std::copy(std::begin(DP_EPP_model_0), std::end(DP_EPP_model_0), par_host->model_par[0]);

		// Numerical stabilization techniques
		par_host->cs_m = 100.0f;
		par_host->cfl = 0.1f;
		par_host->viscosity_type = 1;
		par_host->artificial_viscosity_alpha_m = 0.25f;
		par_host->artificial_pressure_coefficient_material = 0.2f;

		// Simulation output 
		par_host->time_max = 1.0f;
		par_host->time = 0.0f;
		par_host->dt = 1.0e-6f;
		par_host->output_frequency = par_host->time_max / 100.f;

		scalar4* pospres = NULL;;

		std::cout << "Start filling domain with lattice" << std::endl;
		GetParticles(par_host, pospres);
		std::cout << "Lattice completed, in total " << par_host->np << " particles were generated" << std::endl;
		std::cout << par_host->np << " lattice particles are generated" << std::endl;
		std::cout << "Start selecting particles" << std::endl;

		// Boundary
		for (int i = 0; i < par_host->np; i++)
		{
			scalar4 pospres_i = pospres[i];
			if (dim==OneD) pospres_i.x = 0.0f;
			if (dim==OneD || dim==TwoD) pospres_i.y = 0.0f;

			bool bound_left = dim==OneD ? false : pospres_i.x <= bound_thickness;
			bool bound_right = dim==OneD ? false : pospres_i.x >= par_host->domain_size.x - bound_thickness;
			bool bound_bottom = pospres_i.z <= bound_thickness;
			bool bound_front = (dim==OneD || dim==TwoD) ? false : pospres_i.y <= bound_thickness;
			bool bound_back = (dim==OneD || dim==TwoD) ? false : pospres_i.y >= par_host->domain_size.y - bound_thickness;

			if (bound_left || bound_right || bound_bottom || bound_front || bound_back)
			{
				vec_pospres.push_back(pospres_i);
				vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, 0.0f));
				vec_idmassidx.push_back(make_scalar4(0.0f, 0.0, 0.0f, 0.0f));
				vec_stress.push_back(zeros);
				vec_strain.push_back(zeros);

				int blockID;
				status block_status = Fix;
				scalar4 normal_i;
				if (bound_bottom) {
					blockID = 0;
					normal_i = make_scalar4(0.f, 0.f, 1.0f, 1.0f);
				}
				else if (bound_front) {
					blockID = 3;
					normal_i = make_scalar4(0.0f, 1.0f, 0.0f, 1.0f);
				}
				else if (bound_back) {
					blockID = 4;
					normal_i = make_scalar4(0.0f, -1.0f, 0.0f, 1.0f);
				}
				else if (bound_left) {
					blockID = 1;
					normal_i = make_scalar4(1.f, 0.f, 0.0f, 1.0f);
				}
				else {
					blockID = 2;
					normal_i = make_scalar4(-1.f, 0.f, 0.0f, 1.0f);
				}
				vec_codes.push_back(make_int4(blockID, block_status, INT_MIN, INT_MIN));
				vec_normal_m.push_back(normal_i);
				npb++;
			}
		}

		// Material
		for (int i = 0; i < par_host->np; i++)
		{
			scalar4 pospres_i = pospres[i];
			if (dim==OneD) pospres_i.x = 0.0f;
			if (dim==OneD || dim==TwoD) pospres_i.y = 0.0f;

			bool sediment = (dim==OneD ? true : (pospres_i.x >= bound_thickness && pospres_i.x <= par_host->domain_size.x - bound_thickness)) &&
				((dim==OneD || dim==TwoD) ? true : (pospres_i.y >= bound_thickness && pospres_i.y <= par_host->domain_size.y - bound_thickness)) &&
				pospres_i.z >= bound_thickness && pospres_i.z <= soil_height + bound_thickness;

			if (sediment) {
				scalar mass_m = rho0_m * vol;
				vec_codes.push_back(make_int4(6, NonFix, material1, 0));
				vec_pospres.push_back(pospres_i);
				vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, rho0_m));
				vec_idmassidx.push_back(make_scalar4(100.0f, mass_m, 0.0f, 100.0f));
				vec_stress.push_back(zeros);
				vec_strain.push_back(zeros);
				npm++;
			}
		}
		break;
	}

	case 3: { // Inclined Plane Flow	this case is used to valiate the period boundary
		if (dim == TwoD) {
			casename.append("inclined_plane_flow_2D");
			std::cout << "Case: inclined_plane_flow_2D" << std::endl;
		}
		else if (dim == ThreeD) {
			casename.append("inclined_plane_flow_3D");
			std::cout << "Case: inclined_plane_flow_3D" << std::endl;
		}

		// Initial Configuration
		scalar specimen_length = 0.1;
		scalar specimen_width = 1.0;
		scalar specimen_height = 0.4;

		// Domain size
		par_host->periodbound = true;
		par_host->planeleft = make_scalar4(1.0, 0.0, 0.0, 0.0);
		par_host->planeright = make_scalar4(-1.0, 0.0, 0.0, specimen_length);
		par_host->domain_size.x = specimen_length;
		par_host->domain_size.y = dim == TwoD ? 0.f : specimen_width + 2.0f * bound_thickness;
		par_host->domain_size.z = specimen_height + bound_thickness;

		// Material parameters
		scalar theta = 25.0 * PI_D / 180.0;
		par_host->gravity.z = -9.8 * cos(theta); // Gravity
		par_host->gravity.x = 9.8 * sin(theta); // Gravity
		scalar rho0_m = 1500.0f;
		par_host->rho0_m = rho0_m;
		material MuI_DP_EPP_model_0 = { 0 }; // Default MuI_DP_EPP material. You can change it here, or paste the code into the specific case for modification 
		{
			scalar E = 1.5e6f, V = 0.2f, phi1 = 20.0, phi2 = 30.0, I0 = 0.003, rho_g = 2500.0, d_grain = 0.01, lambda = 0.001, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
			GetMaterial_MuI_DP_EPP(E, V, phi1, phi2, I0, rho_g, d_grain, lambda, rho_min, rho_max, MuI_DP_EPP_model_0);
		}
		par_host->n_model = 1;
		model_mat material1 = MuI_DP_EPP;
		std::copy(std::begin(MuI_DP_EPP_model_0), std::end(MuI_DP_EPP_model_0), par_host->model_par[0]); // Herein, DP_EPP_model_0 can be replaced as some other model
		
		// Numerical stabilization techniques
		par_host->cs_m = 100.0f;
		par_host->cfl = 0.1f;
		par_host->viscosity_type = 1;
		par_host->artificial_viscosity_alpha_m = 0.2f;
		par_host->stress_shepard_filter = 0;
		par_host->artificial_pressure_coefficient_material = 0.f;

		// Simulation output 
		par_host->time_max = 5.0f;
		par_host->time = 0.0f;
		par_host->dt = 1.0e-6f;
		par_host->output_frequency = par_host->time_max / 100.f;

		scalar4* pospres = NULL;

		std::cout << "Start filling domain with lattice" << std::endl;
		GetParticles(par_host, pospres);
		std::cout << "Lattice completed, in total " << par_host->np << " particles were generated" << std::endl;
		std::cout << par_host->np << " lattice particles are generated" << std::endl;
		std::cout << "Start selecting particles" << std::endl;

		// Boundary
		for (int i = 0; i < par_host->np; i++)
		{
			scalar4 pospres_i = pospres[i];
			

			bool bound_bottom = pospres_i.z <= bound_thickness;
			bool bound_front =  (pospres_i.y <= bound_thickness);
			bool bound_back = (pospres_i.y >= par_host->domain_size.y - bound_thickness);
			if (dim == TwoD) {
				pospres_i.y = 0.0f;
				bound_front = false;
				bound_back = false;
			}

			if (bound_bottom || bound_front || bound_back)
			{
				vec_pospres.push_back(pospres_i);
				vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, 0.0f));
				vec_idmassidx.push_back(make_scalar4(0.0f, 0.0, 0.0f, 0.0f));
				vec_stress.push_back(zeros);
				vec_strain.push_back(zeros);

				int blockID;
				status block_status = Fix;
				scalar4 normal_i;

				if (bound_bottom) {
					blockID = 0;
					normal_i = make_scalar4(0.f, 0.f, 1.0f, -1.0f);
				}
				else if (bound_front) {
					blockID = 1;
					normal_i = make_scalar4(0.0f, 1.0f, 0.0f, -1.0f);
				}
				else if (bound_back) {
					blockID = 2;
					normal_i = make_scalar4(0.0f, -1.0f, 0.0f, -1.0f);
				}
				vec_codes.push_back(make_int4(blockID, block_status, INT_MIN, INT_MIN));
				vec_normal_m.push_back(normal_i);
				npb++;
			}
		}

		// Material
		for (int i = 0; i < par_host->np; i++)
		{
			scalar K0 = 0.5;
			scalar str_zz = rho0_m * par_host->gravity.z * (-pospres[i].z + bound_thickness + specimen_height);
			tensor2d str_ini = make_tensor2d(str_zz * K0, str_zz * K0, str_zz, 0.0, 0.0, 0.0);
			scalar4 pospres_i = pospres[i];
			if (dim == TwoD) pospres_i.y = 0.0f;

			bool specimen = (dim == TwoD ? true : (pospres_i.y >= bound_thickness && pospres_i.y <= par_host->domain_size.y - bound_thickness)) &&
				pospres_i.z >= bound_thickness && pospres_i.z <= specimen_height + bound_thickness;

			if (specimen) {
				scalar mass_m = rho0_m * vol;
				vec_codes.push_back(make_int4(3, NonFix, material1, 0));
				vec_pospres.push_back(pospres_i);
				vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, rho0_m));
				vec_idmassidx.push_back(make_scalar4(100.0f, mass_m, 0.0f, 100.0f));
				vec_stress.push_back(zeros);
				vec_strain.push_back(zeros);
				npm++;
			}
		}
		break;
	}

	case 4: { // retaining wall			this case is used to valiate the moving boundary
		if (dim == TwoD) {
			casename.append("retaining_wall_2D");
			std::cout << "Case: retaining_wall_2D" << std::endl;
		}

		// Initial Configuration
		scalar box_length = 0.2f;
		scalar box_height = 1.0f;
		scalar box_width = dim == TwoD ? 0.0f : 0.5f;
		scalar moving_wall_height = 1.05f;
		scalar fix_wall_height = 0.04f;

		// Domain size
		par_host->domain_size.x = box_length + fix_wall_height + 2.0f * bound_thickness;
		par_host->domain_size.y = dim == TwoD ? 0.0f : box_width + 2.0f * bound_thickness;
		par_host->domain_size.z = moving_wall_height + fix_wall_height + 2.0 * bound_thickness;

		// Material parameters
		par_host->rho0_m = 1480.0f;
		scalar rho0_m = par_host->rho0_m;
		material DP_EPP_model_0 = { 0 }; // Default DP_EPP material. 
		{
			scalar E = 15.0e6f, V = 0.3f, phi = 30.16f * PI_D / 180.0, psi = 0.0f * PI_D / 180.0, cohesion = 0.0f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
			GetMaterial_DP_EPP(E, V, phi, psi, cohesion, rho_min, rho_max, DP_EPP_model_0);
		}

		material HypoBasic_model_0 = { 0 };  // Default HypoBasic material. 
		{
			scalar Ei = 20.e6f, Vi = 0.3f, phi_c = 35.f * PI_D / 180.0, cohesion = 0.f, p_ref = 1.e5f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
			GetMaterial_HypoBasic(Ei, Vi, phi_c, cohesion, p_ref, rho_min, rho_max, HypoBasic_model_0);
		}

		material HypoCS_model_0 = {0};   // Default HypoCS material. 
		{
			scalar Ei = 5.0e7f, Vi = 0.3f, phi_c = 30.16f * PI_D / 180.0, cohesion = 0.f, p_ref = 1.e5f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;                           // Basic material parameters
			scalar  void_co = 0.7385, lambda = 0.0177, xi = 0.6, alpha = 0.5, void_min = 0.4, void_max = 0.94, void_0 = 0.64, rho0 = par_host->rho0_m;                  // Material parameters related to critical state
			GetMaterial_HypoCS(Ei, Vi, phi_c, cohesion, p_ref, void_co, lambda, xi, alpha, void_min, void_max, void_0, rho0, rho_min, rho_max, HypoCS_model_0);
		}
		par_host->n_model = 1;
		model_mat material1 = HypoCS;
		std::copy(std::begin(HypoCS_model_0), std::end(HypoCS_model_0), par_host->model_par[0]);

		// Numerical stabilization techniques
		par_host->cs_m = 200.0f;
		par_host->cfl = 0.1f;
		par_host->viscosity_type = 1;
		par_host->artificial_viscosity_alpha_m = 1.5f;
		par_host->stress_shepard_filter = 0;
		par_host->artificial_pressure_coefficient_material = 0.01f;

		// Simulation output 
		par_host->time_max = 1.0f;
		par_host->time = 0.0f;
		par_host->dt = 1.0e-6f;
		par_host->output_frequency = par_host->time_max / 100.f;

		scalar4* pospres = NULL;

		std::cout << "Start filling domain with lattice" << std::endl;
		GetParticles(par_host, pospres);
		std::cout << "Lattice completed, in total " << par_host->np << " particles were generated" << std::endl;
		std::cout << par_host->np << " lattice particles are generated" << std::endl;
		std::cout << "Start selecting particles" << std::endl;

		// Boundary
		for (int i = 0; i < par_host->np; i++)
		{
			scalar4 pospres_i = pospres[i];
			if (dim == TwoD) pospres_i.y = 0.0f;

			bool bound_bottom = pospres_i.z <= bound_thickness
				&& pospres_i.x >= fix_wall_height;
			bool bound_bed = (pospres_i.z > fix_wall_height
				&& pospres_i.z <= fix_wall_height + bound_thickness
				&& pospres_i.x >= 0
				&& pospres_i.x <= fix_wall_height)
				|| (pospres_i.z > bound_thickness
					&& pospres_i.z <= fix_wall_height + bound_thickness
					&& pospres_i.x >= fix_wall_height
					&& pospres_i.x <= fix_wall_height + bound_thickness);
			bool bound_left = pospres_i.z > fix_wall_height + bound_thickness
				&& pospres_i.z <= fix_wall_height + bound_thickness + moving_wall_height
				&& pospres_i.x >= fix_wall_height
				&& pospres_i.x <= fix_wall_height + bound_thickness;
			bool bound_right = pospres_i.z >= bound_thickness
				&& pospres_i.z <= fix_wall_height + bound_thickness + moving_wall_height
				&& pospres_i.x >= fix_wall_height + box_length + bound_thickness
				&& pospres_i.x <= fix_wall_height + box_length + 2 * bound_thickness;

			if (bound_bottom || bound_bed || bound_left || bound_right)
			{
				vec_pospres.push_back(pospres_i);
				vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, 0.0f));
				vec_idmassidx.push_back(make_scalar4(0.0f, 0.0, 0.0f, 0.0f));
				vec_stress.push_back(zeros);
				vec_strain.push_back(zeros);

				int blockID;
				int MotionID = INT_MIN;
				status block_status = Fix;
				scalar4 normal_i;

				if (bound_bottom) {
					blockID = 0;
					normal_i = make_scalar4(0.f, 0.f, 1.0f, -1.0f);
				}
				else if (bound_bed) {
					blockID = 1;
					normal_i = make_scalar4(1.0f, 0.0f, 0.0f, -1.0f);
				}
				else if(bound_left){
					blockID = 10;//>10 use contact method
					block_status = NonFix;
					MotionID = 0;
					normal_i = make_scalar4(1.f, 0.f, 0.0f, 1.0f);
				}
				else if (bound_right) {
					blockID = 11;//>10 use contact method
					normal_i = make_scalar4(-1.f, 0.f, 0.0f, 1.0f);
				}
				vec_codes.push_back(make_int4(blockID, block_status, INT_MIN, MotionID));
				vec_normal_m.push_back(normal_i);
				npb++;
			}
		}

		// Material
		for (int i = 0; i < par_host->np; i++)
		{
			scalar4 pospres_i = pospres[i];
			scalar K0 = 0.5;
			scalar str_zz = rho0_m * par_host->gravity.z * (-pospres[i].z + bound_thickness + box_height + fix_wall_height);
			tensor2d str_ini = make_tensor2d(str_zz * K0, str_zz * K0, str_zz, 0.0, 0.0, 0.0);
			if (dim == TwoD) pospres_i.y = 0.0f;

			bool specimen = (pospres_i.z > bound_thickness)
				&& (pospres_i.z <= box_height + fix_wall_height + bound_thickness)
				&& (pospres_i.x > fix_wall_height + bound_thickness)
				&& (pospres_i.x <= fix_wall_height + bound_thickness + box_length);
			if (specimen) {
				scalar mass_m = rho0_m * vol;
				vec_codes.push_back(make_int4(6, NonFix, material1, 0));
				vec_pospres.push_back(pospres_i);
				vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, rho0_m));
				vec_idmassidx.push_back(make_scalar4(100.0f, mass_m, 0.0f, 100.0f));
				vec_stress.push_back(str_ini);
				vec_strain.push_back(zeros);
				npm++;
			}
		}
		cudaFreeHost(pospres);

		// Define the number of moving rigid body
		const int n_moving_rigid_body = 1;
		par_host->n_motion = n_moving_rigid_body;
		MotionTable moving_rb[n_moving_rigid_body];
		
		// Define the moving details for each rigid body
		// Tanslation part	
		moving_rb[0].is_translation = false;
		//moving_rb[0].trans_vel = make_scalar3(0.0, 0.0, 0.0);
		moving_rb[0].table_size_trans = 1;
		//moving_rb[0].trans_time[0] = {0.2};
		moving_rb[0].trans_time[0] = {par_host->time_max};
		//moving_rb[0].trans_acc[0] = make_scalar3(0.0, 0.0, 0.0);
		//moving_rb[0].trans_vel[0] = make_scalar3(0.0, 0.0, 0.0);
		moving_rb[0].trans_acc[0] = make_scalar3(0.0, 0.0, 0.0);
		moving_rb[0].trans_vel[0] = make_scalar3(-0.005, 0.0, 0.0);
		// Rotation part
		moving_rb[0].is_rotation = true;
		moving_rb[0].pattern = Cuboid;
		moving_rb[0].pos_rp = make_scalar3(fix_wall_height + bound_thickness, 0, fix_wall_height + bound_thickness);
		moving_rb[0].rotat_vel = make_scalar3(0.0, 0.01, 0.0);
		moving_rb[0].table_size_rotat = 1;
		//moving_rb[0].rotat_time[0] = { 1.0 };
		moving_rb[0].rotat_time[0] = { par_host->time_max };
		//moving_rb[0].rotat_acc[0] = make_scalar3(0.0, -0.075, 0.0);
		moving_rb[0].rotat_acc[0] = make_scalar3(0.0, 0.0, 0.0);
		for (int i = 0; i < par_host->n_motion; i++) vec_mt.push_back(moving_rb[i]);
		
		break;
	}

	case 5: { // buried pipe			this case is used to valiate the complex geometry of nonlocal contact
		if (dim == TwoD) {
			casename.append("buried_pipe");
			std::cout << "Case: buried_pipe_2D" << std::endl;
		}

		// Initial Configuration
		scalar box_length = 0.75f;
		scalar box_height = 0.45f;
		scalar box_width = dim == TwoD ? 0.0f : 0.5f;
		scalar D_pipe = 0.075f;
		scalar pipe_center_x = (box_length + bound_thickness)/2;
		scalar pipe_center_z = (box_height + bound_thickness) - 1.5 * D_pipe;

		// Domain size
		par_host->domain_size.x = box_length + 2.0f * bound_thickness;
		par_host->domain_size.y = dim == TwoD ? 0.0f : box_width + 2.0f * bound_thickness;
		par_host->domain_size.z = box_height + 2.0f * D_pipe + 2.0f * bound_thickness;

		// Material parameters
		par_host->rho0_m = 1734.7f;
		scalar rho0_m = par_host->rho0_m;
		material DP_EPP_model_0 = { 0 }; // Default DP_EPP material. 
		{
			scalar E = 15.0e6f, V = 0.2f, phi = 32.0f * PI_D / 180.0, psi = 16.0f * PI_D / 180.0, cohesion = 0.0f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
			GetMaterial_DP_EPP(E, V, phi, psi, cohesion, rho_min, rho_max, DP_EPP_model_0);
		}

		material HypoBasic_model_0 = { 0 };  // Default HypoBasic material. 
		{
			scalar Ei = 20.e6f, Vi = 0.2f, phi_c = 32.f * PI_D / 180.0, cohesion = 0.f, p_ref = 1.e5f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;
			GetMaterial_HypoBasic(Ei, Vi, phi_c, cohesion, p_ref, rho_min, rho_max, HypoBasic_model_0);
		}

		material HypoCS_model_0 = {0};   // Default HypoCS material. 
		{
			scalar Ei = 10.0e7f, Vi = 0.2f, phi_c = 32.f * PI_D / 180.0, cohesion = 0.f, p_ref = 1.e5f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;                           // Basic material parameters
			scalar  void_co = 0.925, lambda = 0.037, xi = 1.0, alpha = 0.4, void_min = 0.51, void_max = 0.79, void_0 = 0.5324, rho0 = par_host->rho0_m;                  // Material parameters related to critical state
			GetMaterial_HypoCS(Ei, Vi, phi_c, cohesion, p_ref, void_co, lambda, xi, alpha, void_min, void_max, void_0, rho0, rho_min, rho_max, HypoCS_model_0);
		}/*
		material HypoCS_model_0 = { 0 };   // Default HypoCS material. 
		{
			scalar Ei = 1.5e7f, Vi = 0.2f, phi_c = 32.f * PI_D / 180.0, cohesion = 0.f, p_ref = 1.e5f, rho_min = rho0_m * 0.8, rho_max = rho0_m * 1.2;                           // Basic material parameters
			scalar  void_co = 0.925, lambda = 0.037, xi = 1.0, alpha = 0.5, void_min = 0.51, void_max = 0.79, void_0 = 0.7368, rho0 = par_host->rho0_m;                  // Material parameters related to critical state
			GetMaterial_HypoCS(Ei, Vi, phi_c, cohesion, p_ref, void_co, lambda, xi, alpha, void_min, void_max, void_0, rho0, rho_min, rho_max, HypoCS_model_0);
		}*/
		par_host->n_model = 1;
		model_mat material1 = HypoCS;
		std::copy(std::begin(HypoCS_model_0), std::end(HypoCS_model_0), par_host->model_par[0]);

		// Numerical stabilization techniques
		par_host->cs_m = 200.0f;
		par_host->cfl = 0.1f;
		par_host->viscosity_type = 1;
		par_host->artificial_viscosity_alpha_m = 1.0f;
		par_host->stress_shepard_filter = 0;
		par_host->artificial_pressure_coefficient_material = 0.01f;

		// Simulation output 
		par_host->time_max = 2.0f;
		par_host->time = 0.0f;
		par_host->dt = 1.0e-6f;
		par_host->output_frequency = par_host->time_max / 100.f;

		scalar4* pospres = NULL;

		std::cout << "Start filling domain with lattice" << std::endl;
		GetParticles(par_host, pospres);
		std::cout << "Lattice completed, in total " << par_host->np << " particles were generated" << std::endl;
		std::cout << par_host->np << " lattice particles are generated" << std::endl;
		std::cout << "Start selecting particles" << std::endl;

		// Boundary
		for (int i = 0; i < par_host->np; i++)
		{
			scalar4 pospres_i = pospres[i];
			if (dim == TwoD) pospres_i.y = 0.0f;

			bool bound_left = dim == OneD ? false : pospres_i.x <= bound_thickness && pospres_i.z > bound_thickness;
			bool bound_right = dim == OneD ? false : pospres_i.x >= par_host->domain_size.x - bound_thickness && pospres_i.z > bound_thickness;
			bool bound_bottom = pospres_i.z <= bound_thickness;
			bool bound_front = dim == TwoD ? false : pospres_i.y <= bound_thickness;
			bool bound_back = dim == TwoD ? false : pospres_i.y >= par_host->domain_size.y - bound_thickness;
			bool pipe = ((pospres_i.x - pipe_center_x) * (pospres_i.x - pipe_center_x) + (pospres_i.z - pipe_center_z) * (pospres_i.z - pipe_center_z)) <= (D_pipe / 2) * (D_pipe / 2)
				&& ((pospres_i.x - pipe_center_x) * (pospres_i.x - pipe_center_x) + (pospres_i.z - pipe_center_z) * (pospres_i.z - pipe_center_z)) >= (D_pipe / 4) * (D_pipe / 4);

			if (bound_left || bound_right || bound_bottom || bound_front || bound_back || pipe)
			{
				vec_pospres.push_back(pospres_i);
				vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, 0.0f));
				vec_idmassidx.push_back(make_scalar4(0.0f, 0.0, 0.0f, 0.0f));
				vec_stress.push_back(zeros);
				vec_strain.push_back(zeros);

				int blockID;
				int MotionID = INT_MIN;
				status block_status = Fix;
				scalar4 normal_i;

				if (bound_bottom) {
					blockID = 0;
					normal_i = make_scalar4(0.f, 0.f, 1.0f, 1.0f);
				}
				else if (bound_front) {
					blockID = 4;
					normal_i = make_scalar4(0.0f, 1.0f, 0.0f, 1.0f);
				}
				else if (bound_back) {
					blockID = 5;
					normal_i = make_scalar4(0.0f, -1.0f, 0.0f, 1.0f);
				}
				else if (bound_left) {
					blockID = 1;
					normal_i = make_scalar4(0.f, 0.f, 1.0f, 1.0f);
				}
				else if (bound_right) {
					blockID = 2;
					normal_i = make_scalar4(-1.f, 0.f, 0.0f, 1.0f);
				}
				else if (pipe) {
					blockID = 10;//>10 use contact method
					block_status = NonFix;
					MotionID = 0;
					normal_i = make_scalar4(0.f, 0.f, 1.0f, 1.0f);
				}
				vec_codes.push_back(make_int4(blockID, block_status, INT_MIN, MotionID));
				vec_normal_m.push_back(normal_i);
				npb++;
			}
		}

		// Material
		for (int i = 0; i < par_host->np; i++)
		{
			scalar4 pospres_i = pospres[i];
			scalar K0 = 0.5;
			scalar str_zz = rho0_m * par_host->gravity.z * (-pospres[i].z + bound_thickness + box_height);
			tensor2d str_ini = make_tensor2d(str_zz * K0, str_zz * K0, str_zz, 0.0, 0.0, 0.0);
			if (dim == TwoD) pospres_i.y = 0.0f;

			bool soil = pospres_i.x >= bound_thickness && pospres_i.x <= bound_thickness + box_length &&
				(dim == TwoD ? true : (pospres_i.y >= bound_thickness && pospres_i.y <= par_host->domain_size.y - bound_thickness)) &&
				pospres_i.z >= bound_thickness && pospres_i.z <= bound_thickness + box_height;
			bool pipe_m = ((pospres_i.x - pipe_center_x) * (pospres_i.x - pipe_center_x) + (pospres_i.z - pipe_center_z) * (pospres_i.z - pipe_center_z)) <= (D_pipe / 2) * (D_pipe / 2);
			if (pipe_m) soil = 0;

			if (soil) {
				scalar mass_m = rho0_m * vol;
				vec_codes.push_back(make_int4(6, NonFix, material1, 0));
				vec_pospres.push_back(pospres_i);
				vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, rho0_m));
				vec_idmassidx.push_back(make_scalar4(100.0f, mass_m, 0.0f, 100.0f));
				vec_stress.push_back(str_ini);
				vec_strain.push_back(zeros);
				npm++;
			}
		}
		cudaFreeHost(pospres);

		// Define the number of moving rigid body
		const int n_moving_rigid_body = 1;
		par_host->n_motion = n_moving_rigid_body;
		MotionTable moving_rb[n_moving_rigid_body];

		// Define the moving details for each rigid body
		// Tanslation part	
		moving_rb[0].is_translation = true;
		//moving_rb[0].trans_vel = make_scalar3(0.0, 0.0, 0.0);
		moving_rb[0].table_size_trans = 1;
		//moving_rb[0].trans_time[0] = {0.0};
		moving_rb[0].trans_time[0] = { par_host->time_max };
		//moving_rb[0].trans_acc[0] = make_scalar3(0.0, 0.0, 0.0);
		//moving_rb[0].trans_vel[0] = make_scalar3(0.0, 0.0, 0.0);
		moving_rb[0].trans_acc[0] = make_scalar3(0.0, 0.0, 0.0);
		moving_rb[0].trans_vel[0] = make_scalar3(0.0, 0.0, 0.01);
		// Rotation part
		moving_rb[0].is_rotation = false;
		moving_rb[0].pattern = Cuboid;
		moving_rb[0].pos_rp = make_scalar3(0, 0, 0);
		moving_rb[0].rotat_vel = make_scalar3(0.0, 0.1, 0.0);
		moving_rb[0].table_size_rotat = 1;
		//moving_rb[0].rotat_time[0] = { 1.0 };
		moving_rb[0].rotat_time[0] = { par_host->time_max };
		//moving_rb[0].rotat_acc[0] = make_scalar3(0.0, -0.075, 0.0);
		moving_rb[0].rotat_acc[0] = make_scalar3(0.0, 0.0, 0.0);
		for (int i = 0; i < par_host->n_motion; i++) vec_mt.push_back(moving_rb[i]);

		break;
	}

	case 6: { // Rotating Drum		
		casename.append("Rotating_Drum_2D");
		std::cout << "Case: Rotating_Drum_2D" << std::endl;

		// Initial Configuration
		scalar diameter = 1.0;
		scalar filling_level = 0.35;
		scalar rotation_rate = 10.0; //unit: rpm
		scalar box_length = diameter;
		scalar box_height = diameter;
		scalar radius_int = 0.5 * diameter;
		scalar center_x = radius_int + bound_thickness;
		scalar center_z = radius_int + bound_thickness;
		scalar sand_depth = diameter * filling_level;

		// Domain size
		par_host->domain_size.x = box_length + 2.0 * bound_thickness;
		par_host->domain_size.y = 0.0;
		par_host->domain_size.z = box_height + 2.0 * bound_thickness;

		// Material parameters
		par_host->rho0_m = 1.5e3; //kg*m^-3
		scalar rho0_m = par_host->rho0_m;

		// Numerical stabilization techniques
		par_host->stress_shepard_filter = 30;
		par_host->cs_m = 50.0; //
		par_host->cfl = 0.2;
		par_host->viscosity_type = 1;
		par_host->artificial_viscosity_alpha_m = 0.1;
		par_host->artificial_pressure_coefficient_material = 0.0; // artificial pressure has little effect on the profile but can garantee SPH particles inside the calculation domain

		// Simulation output 
		par_host->time_max = 2.0;
		par_host->time = 0.0;
		par_host->dt = 1.0e-6;
		par_host->output_frequency = par_host->time_max / 100.0;
		par_host->output_format = 2; // Vtu

		scalar4* pospres = NULL;
		std::cout << "Start filling domain with lattice" << std::endl;
		GetParticles(par_host, pospres);
		std::cout << "Lattice completed, in total " << par_host->np << " particles were generated" << std::endl;
		std::cout << par_host->np << " lattice particles are generated" << std::endl;
		std::cout << "Start selecting particles" << std::endl;

		// Boundary
		scalar ang_vel = 2.0 * PI_D * rotation_rate / 60.0;
		for (int i = 0; i < ceil(2.0 * hdr); i++) {
			scalar radius = static_cast<scalar>(radius_int + dr * i);
			scalar perimeter = 2.0 * PI_D * radius;
			int number_in_layer = 2 * (perimeter / static_cast<scalar>(dr) / 2);
			scalar interval = 2.0 * PI_D / number_in_layer;
			for (int j = 0; j < number_in_layer; j++) {
				scalar4 pospres_i = make_scalar4(0.0);
				scalar theta = interval * j;
				pospres_i.x = radius * cos(theta) + center_x;
				pospres_i.y = 0.f;
				pospres_i.z = radius * sin(theta) + center_z;
				vec_codes.push_back(make_int4(0, NonFix, INT_MIN, 0));
				vec_pospres.push_back(pospres_i);
				vec_idmassidx.push_back(make_scalar4(0.0));
				vec_stress.push_back(zeros);
				vec_strain.push_back(zeros);
				vec_normal_m.push_back(make_scalar4(-cos(theta), 0.0, -sin(theta), -1.0));
				vec_velrhop.push_back(make_scalar4(0.0));
				npb++;
			}
		}

		// Material
		int number_layer = radius_int / dr;
		scalar interval_layer = radius_int / number_layer;
		scalar3 center = make_scalar3(center_x, 0.f, center_z);
		for (int i = 0; i < number_layer; i++) {
			scalar radius = interval_layer * i;
			scalar perimeter = 2.0 * PI_D * radius;
			int number_in_layer = perimeter / static_cast<scalar>(dr) < 1 ? 1 : 2 * int(perimeter / static_cast<scalar>(dr) / 2);
			scalar interval_theta = 2.0 * PI_D / number_in_layer;
			for (int j = 0; j < number_in_layer; j++) {
				scalar4 pospres_i = make_scalar4(0.0);
				scalar theta = interval_theta * j + interval_theta * 0.5;
				pospres_i.x = radius * cos(theta) + center_x;
				pospres_i.y = 0.0;
				pospres_i.z = radius * sin(theta) + center_z;
				if (pospres_i.z <= sand_depth + bound_thickness + 0.5 * dr) {
					scalar mass_m = rho0_m * vol;
					scalar dist_to_center = EulerianNorm(make_scalar3(pospres_i.x - center_x, 0.0, pospres_i.z - center_z));
					scalar ID = pospres_i.x < center_x + 0.5 * dr ? 101.f : 100.f;
					scalar conc = pospres_i.x < center_x + 0.5 * dr ? 1.f : 0.f;
					vec_codes.push_back(make_int4(1, NonFix, material1, 0));
					vec_pospres.push_back(pospres_i);
					vec_velrhop.push_back(make_scalar4(0.0f, 0.0f, 0.0f, rho0_m));
					vec_idmassidx.push_back(make_scalar4(100.f, mass_m, 0.0f, ID));
					vec_stress.push_back(zeros);
					vec_strain.push_back(zeros);
					npm++;
				}
			}
		}
		cudaFreeHost(pospres);
		
		// Define the number of moving rigid body
		const int n_moving_rigid_body = 1;
		par_host->n_motion = n_moving_rigid_body;
		MotionTable moving_rb[n_moving_rigid_body];

		// Define the moving details for each rigid body
		// Tanslation part	
		moving_rb[0].is_translation = false;

		// Rotation part
		moving_rb[0].is_rotation = true;
		moving_rb[0].pattern = Cuboid;
		moving_rb[0].pos_rp = make_scalar3(center_x, 0, center_z);
		moving_rb[0].rotat_vel = make_scalar3(0.0, ang_vel, 0.0);
		moving_rb[0].table_size_rotat = 0;
		for (int i = 0; i < par_host->n_motion; i++) vec_mt.push_back(moving_rb[i]);

		break;
	}

	default: {
		std::cout << "Error! Undefined case number!" << std::endl;
		break;
	}
	}

	par_host->npb = npb;
	par_host->npm = npm;
	par_host->np = npb + npm;

	std::cout << "Finish selecting partcles" << std::endl;
	std::cout << "Selected particles: total number -    " << par_host->np << std::endl;
	std::cout << "                    boundary number - " << par_host->npb << std::endl;
	std::cout << "                    material number - " << par_host->npm << std::endl;

	std::cout << "Start saving the dat file" << std::endl;
	SaveDatFile(casename, par_host, vec_codes, vec_pospres, vec_velrhop, vec_idmassidx, vec_stress, vec_strain, vec_normal_m);
	std::cout << "Finish saving the dat file" << std::endl;

	std::cout << "Start saving the xml file" << std::endl;
	SaveXmlFile(casename, par_host);
	std::cout << "Finish saving the xml file" << std::endl;

	if (par_host->n_motion > 0) {
		std::cout << "Start saving the motion file" << std::endl;
		SaveMotionFile(casename, par_host, vec_mt);
		std::cout << "Finish saving the motion file" << std::endl;
	}
}