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

#include "loquat.h"
//#include "adami_boundary.h"
//#include "particle_interaction.h"
//#include "predictor_corrector.h"
//#include "field_smoothing.h"
#include "utilities.h"
#include <helper_cuda.h>

LoquatSph::LoquatSph(std::string name)
{
	project_name = name;
    io = NULL;
    grid = NULL;
    par_host = NULL;
    par_device = NULL;
    timer = NULL;
	moving_body = NULL;
	kinematic_table = NULL;
    selected_device = 0;

	codes = NULL;
	cellpos = NULL;
	pospres = NULL;
	pospres_ori = NULL;
	velrhop = NULL;
	idmassidx = NULL;
	str = NULL;
	stn = NULL;
	normal_m = NULL;
	contact_mag = NULL;
	contact_var = NULL;
		
	pospres_ini = NULL;
	velrhop_ini = NULL;
	str_ini = NULL;
	stn_ini = NULL;
		
	accad = NULL;
	spin = NULL;
	acestn = NULL;
	
	int4_cache = NULL;
	scalar_cache = NULL;
	scalar4_cache = NULL;
	tensor2d_cache = NULL;

	io = new LoquatIo(project_name);
	grid = new LoquatGrid();
}


LoquatSph::~LoquatSph()
{
	if (codes != NULL) cudaFree(codes);
	if(cellpos != NULL) cudaFree(cellpos);
	if(pospres != NULL) cudaFree(pospres);
	if(pospres_ori != NULL) cudaFree(pospres_ori);
	if(velrhop != NULL) cudaFree(velrhop);
	if(idmassidx != NULL) cudaFree(idmassidx);
	if(str != NULL) cudaFree(str);
	if(stn != NULL) cudaFree(stn);
	if(normal_m != NULL) cudaFree(normal_m);
	if (contact_mag != NULL) cudaFree(contact_mag);
	if (contact_var != NULL) cudaFree(contact_var);

	if(pospres_ini != NULL) cudaFree(pospres_ini);
	if(velrhop_ini != NULL) cudaFree(velrhop_ini);
	if(str_ini != NULL) cudaFree(str_ini);
	if(str_ini != NULL) cudaFree(stn_ini);

	if(accad != NULL) cudaFree(accad);
    if(spin != NULL) cudaFree(spin);
	if(acestn != NULL) cudaFree(acestn);

	if(scalar_cache != NULL) cudaFree(scalar_cache);
	if (int4_cache != NULL) cudaFree(int4_cache);
	if(scalar4_cache != NULL) cudaFree(scalar4_cache);
	if(tensor2d_cache != NULL) cudaFree(tensor2d_cache);
}


std::string LoquatSph::TimeToString(double time)
{
	int days = 0;
	int hours = 0;
	int minutes = 0;
	int seconds = static_cast<int>(time);

	hours = seconds / 3600;
	if(hours > 24) 
	{
		days = hours / 24;
		hours = (seconds - days * 24 * 3600) / 3600;
	}

	minutes = (seconds % 3600) / 60;
	seconds = (seconds % 3600) % 60;

	std::stringstream string;

	string << std::setfill('0') << std::setw(2) << days << "d ";
	string << std::setfill('0') << std::setw(2) << hours << "h ";
	string << std::setfill('0') << std::setw(2) << minutes << "m ";
	string << std::setfill('0') << std::setw(2) << seconds << "s ";

	return string.str();
}


void LoquatSph::InitializeParameters()
{
	par_host->h = par_host->dr * par_host->hdr;
	par_host->i_h = 1.0f / par_host->h;
	par_host->eta = 0.01f * par_host->h * par_host->h;
	par_host->bound_thickness = ceil(2.0 *  par_host->hdr) * par_host->dr;
	
	scalar h_d = static_cast<scalar>(par_host->h);

	if (par_host->dim == OneD) {
		par_host->vol_def = par_host->dr;
		par_host->cell_num.x = 1;
		par_host->cell_num.y = 1;
		scalar kernel_normalization_par = 3.0 / (32 * h_d);
		par_host->kernel_normalization_par = static_cast<scalar>(kernel_normalization_par);
		scalar kernel_gradient_normalization_par = -15.0 / (32 * pow(h_d, 3));
		par_host->kernel_gradient_normalization_par = static_cast<scalar>(kernel_gradient_normalization_par);
	}
	else if (par_host->dim == TwoD){
		par_host->vol_def = pow(par_host->dr, 2);
		par_host->cell_num.y = 1;
		par_host->cell_num.x = static_cast<int>(0.5f * par_host->i_h * par_host->domain_size.x) + 1;

		scalar kernel_normalization_par = 7.0 / (32.0 * PI_D * pow(h_d, 2));
		par_host->kernel_normalization_par = static_cast<scalar>(kernel_normalization_par);
		scalar kernel_gradient_normalization_par = -35.0 / (32 * PI_D * pow(h_d, 4));
		par_host->kernel_gradient_normalization_par = static_cast<scalar>(kernel_gradient_normalization_par);
	}
	else if (par_host->dim == ThreeD) {
		par_host->vol_def = pow(par_host->dr, 3);
		par_host->cell_num.x = static_cast<int>(0.5f * par_host->i_h * par_host->domain_size.x) + 1;
		par_host->cell_num.y = static_cast<int>(0.5f * par_host->i_h * par_host->domain_size.y) + 1;

		scalar kernel_normalization_par = 42.0 / (256.0 * PI_D * pow(h_d, 3));
		par_host->kernel_normalization_par = static_cast<scalar>(kernel_normalization_par);
		scalar kernel_gradient_normalization_par = -5.0 * 42 / (256.0 * PI_D * pow(h_d, 5));
		par_host->kernel_gradient_normalization_par = static_cast<scalar>(kernel_gradient_normalization_par);
	}
	else {
		std::cout << "The input dimension of problem is " << par_host->dim << std::endl;
		std::cout << "Check the dimension of problem!" << std::endl;
		exit(EXIT_FAILURE);
	}
	par_host->cell_num.z = static_cast<int>(0.5f * par_host->i_h * par_host->domain_size.z) + 1;
	int ncell_one_layer = par_host->cell_num.x * par_host->cell_num.y * par_host->cell_num.z;

	par_host->ncell_material = ncell_one_layer;
	par_host->ncell_delete_m = 2 * ncell_one_layer;
	par_host->ncell = par_host->ncell_delete_m + 1;
	par_host->np_delete_m = 0;

	printf("ncellx = %d, ncelly = %d, ncellz = %d, ncell = %d\n", par_host->cell_num.x, par_host->cell_num.y, par_host->cell_num.z, par_host->ncell);
	printf("ncell_material = %d  \n", par_host->ncell_material);
	printf("ncell_delete_m = %d  \n", par_host->ncell_delete_m);
	printf("ncell = %d  \n", par_host->ncell);
	printf("\n");

	scalar q = par_host->dr * par_host->i_h;
	par_host->kernel_zero = par_host->kernel_normalization_par * (2.0f) * (2.0f) * (2.0f) * (2.0f) * (0.5f);
	par_host->kernel_dr = par_host->kernel_normalization_par * (2.0f - q) * (2.0f - q) * (2.0f - q) * (2.0f - q) * (q + 0.5f);
}


void LoquatSph::AllocateGpuMemory(){
    checkCudaErrors(cudaMalloc((void **)&par_device, sizeof(Parameters)));

	checkCudaErrors(cudaMalloc((void **)&codes, par_host->np * sizeof(int4)));
	checkCudaErrors(cudaMalloc((void **)&cellpos, par_host->np * sizeof(int4)));
    checkCudaErrors(cudaMalloc((void **)&pospres, par_host->np * sizeof(scalar4)));
	checkCudaErrors(cudaMalloc((void **)&pospres_ori, par_host->np * sizeof(scalar4)));
    checkCudaErrors(cudaMalloc((void **)&velrhop, par_host->np * sizeof(scalar4)));
	checkCudaErrors(cudaMalloc((void **)&idmassidx, par_host->np * sizeof(scalar4)));
    checkCudaErrors(cudaMalloc((void **)&pospres_ini, par_host->np * sizeof(scalar4)));
    checkCudaErrors(cudaMalloc((void **)&velrhop_ini, par_host->np * sizeof(scalar4)));   
    checkCudaErrors(cudaMalloc((void **)&accad, par_host->np * sizeof(scalar4)));
    checkCudaErrors(cudaMalloc((void **)&scalar_cache, par_host->np * sizeof(scalar)));
	checkCudaErrors(cudaMalloc((void **)&int4_cache, par_host->np * sizeof(int4)));
    checkCudaErrors(cudaMalloc((void **)&scalar4_cache, par_host->np * sizeof(scalar4)));
	checkCudaErrors(cudaMalloc((void **)&normal_m, par_host->npb * sizeof(scalar4)));
	checkCudaErrors(cudaMalloc((void **)&contact_mag, par_host->np * sizeof(tensor2dful)));
	checkCudaErrors(cudaMalloc((void **)&contact_var, par_host->np * sizeof(scalar4)));

	checkCudaErrors(cudaMalloc((void **)&str, (par_host->npb + par_host->npm) * sizeof(tensor2d)));
	checkCudaErrors(cudaMalloc((void **)&stn, (par_host->npb + par_host->npm) * sizeof(tensor2d)));
	checkCudaErrors(cudaMalloc((void **)&str_ini, (par_host->npb + par_host->npm) * sizeof(tensor2d)));
	checkCudaErrors(cudaMalloc((void **)&stn_ini, (par_host->npb + par_host->npm) * sizeof(tensor2d)));		
	checkCudaErrors(cudaMalloc((void **)&spin, (par_host->npb + par_host->npm) * sizeof(scalar4)));
	checkCudaErrors(cudaMalloc((void **)&acestn, (par_host->npb + par_host->npm) * sizeof(tensor2d)));		
	checkCudaErrors(cudaMalloc((void **)&tensor2d_cache, (par_host->npb + par_host->npm) * sizeof(tensor2d)));

}


void LoquatSph::CopyDataFromHostToDevice(){
	checkCudaErrors(cudaMemcpy(codes, io->GetCodes(), par_host->np * sizeof(int4), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pospres, io->GetPosPres(), par_host->np * sizeof(scalar4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pospres_ori, io->GetPosPres(), par_host->np * sizeof(scalar4), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(velrhop, io->GetVelRhop(), par_host->np * sizeof(scalar4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(idmassidx, io->GetIdMassIdx(), par_host->np * sizeof(scalar4), cudaMemcpyHostToDevice));
 	checkCudaErrors(cudaMemcpy(normal_m, io->GetNormalMaterial(), par_host->npb * sizeof(scalar4), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(str, io->GetStr(), (par_host->npb + par_host->npm) * sizeof(tensor2d), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(stn, io->GetStn(), (par_host->npb + par_host->npm) * sizeof(tensor2d), cudaMemcpyHostToDevice));

}


void LoquatSph::CopyDataFromDeviceToHost(){
	checkCudaErrors(cudaMemcpy(io->GetCodes(), codes, par_host->np * sizeof(int4), cudaMemcpyDeviceToHost));
	grid->PositionFromRelative2Absolute(par_host, par_device, cellpos, pospres, scalar4_cache);
    checkCudaErrors(cudaMemcpy(io->GetPosPres(), scalar4_cache, par_host->np * sizeof(scalar4), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(io->GetVelRhop(), velrhop, par_host->np * sizeof(scalar4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetAccad(), accad, par_host->np * sizeof(scalar4), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(io->GetIdMassIdx(), idmassidx, par_host->np * sizeof(scalar4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetNormalMaterial(), normal_m, par_host->npb * sizeof(scalar4), cudaMemcpyDeviceToHost));

	StressSavingRegularization();
	checkCudaErrors(cudaMemcpy(io->GetStr(), tensor2d_cache, (par_host->npb + par_host->npm) * sizeof(tensor2d), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(io->GetStn(), stn, (par_host->npb + par_host->npm) * sizeof(tensor2d), cudaMemcpyDeviceToHost));

	memcpy(io->GetParameters(), par_host, sizeof(Parameters));
}


void LoquatSph::SortArraysPrimary(){
	grid->SortArrayInt4(par_host->np, codes, int4_cache);
	grid->SortArrayInt4(par_host->np, cellpos, int4_cache);
	grid->SortArrayScalar4(par_host->np, pospres, scalar4_cache);
	grid->SortArrayScalar4(par_host->np, pospres_ori, scalar4_cache);
	grid->SortArrayScalar4(par_host->np, velrhop, scalar4_cache);
	grid->SortArrayScalar4(par_host->np, idmassidx, scalar4_cache);
	grid->SortArrayScalar4(par_host->npb, normal_m, scalar4_cache);

	grid->SortArrayTensor2d((par_host->npb + par_host->npm), str, tensor2d_cache);
	grid->SortArrayTensor2d((par_host->npb + par_host->npm), stn, tensor2d_cache);

}


void LoquatSph::SortArraysAll(){
	grid->SortArrayInt4(par_host->np, codes, int4_cache);
	grid->SortArrayInt4(par_host->np, cellpos, int4_cache);
	grid->SortArrayScalar4(par_host->np, pospres, scalar4_cache);
	grid->SortArrayScalar4(par_host->np, pospres_ori, scalar4_cache);
	grid->SortArrayScalar4(par_host->np, velrhop, scalar4_cache);
	grid->SortArrayScalar4(par_host->np, idmassidx, scalar4_cache);
	
	grid->SortArrayScalar4(par_host->np, pospres_ini, scalar4_cache);
	grid->SortArrayScalar4(par_host->np, velrhop_ini, scalar4_cache);
	
	if(par_host->npm != 0){
		grid->SortArrayTensor2d((par_host->npb + par_host->npm), str, tensor2d_cache);
		grid->SortArrayTensor2d((par_host->npb + par_host->npm), stn, tensor2d_cache);

		grid->SortArrayTensor2d((par_host->npb + par_host->npm), str_ini, tensor2d_cache);
		grid->SortArrayTensor2d((par_host->npb + par_host->npm), stn_ini, tensor2d_cache);
	}
}

void LoquatSph::GPUInfoExibition(int argc, char* argv[])
{
    std::cout << "**********************************" << std::endl;
	std::cout << "** LOQUAT Solver for single GPU **" << std::endl;
	std::cout << "**********************************" << std::endl;
	std::cout << " " << std::endl;

	switch(argc) {
		case 1:	{
			std::cout << "Error! Please give the project name!" << std::endl;
			exit(EXIT_FAILURE);
			break;
		}
		
		case 2:	{
			selected_device = 0;
			break;
		}
		
		case 3:	{
			selected_device = atoi(argv[2]);
			break;
		}
		
		default:
			break;
	}

	std::string path(project_name);
	if(!boost::filesystem::exists(path))
	{
		std::cout << "Create results folder: " << path << std::endl;
		boost::filesystem::create_directory(path);
	}

	int number_of_gpus;
	cudaGetDeviceCount(&number_of_gpus);
	std::cout << "Number of GPUs on this computer: " << number_of_gpus << std::endl;
	std::cout << "Information of the GPUs: " << std::endl;
	for(int i = 0; i < number_of_gpus; i++)
	{
		std::stringstream info;
		info << "  (" << i << ") ";
		cudaDeviceProp gpu_properties;
		cudaGetDeviceProperties(&gpu_properties, i);
		info << gpu_properties.name << " " << gpu_properties.totalGlobalMem / 1048576 << " MB"
			 << " (" << gpu_properties.major << "." << gpu_properties.minor << ")";
		std::cout << info.str() << std::endl;
	}
	std::cout << "Device check finished" << std::endl;

	cudaSetDevice(selected_device);
	std::cout << std::endl;
	std::cout << "Simulation running on GPU " << selected_device << std::endl;
}

void LoquatSph::TimerInitiation()
{
	timer = new Timer;
	timer->total = 0.0;
	timer->neighbor_search = 0.0;
	timer->adami_boundary = 0.0;
	timer->interpolation = 0.0;
	timer->interaction = 0.0;
	timer->integration = 0.0;
	timer->saving = 0.0;

	timer->clock_initial = std::clock();
	timer->clock_interval = std::clock();

	time = static_cast<double>(par_host->time);
	step = 0;
	output_time = 0.0;
	step_interval = 0;
}

// Calculate step interval and print output header
void LoquatSph::StepCalculation()
{
	scalar dt_m = par_host->h / par_host->cs_m;
	par_host->dt = par_host->cfl * dt_m;
	par_host->time = static_cast<scalar>(time);
	checkCudaErrors(cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice));	
	printf(" SimTime = %0.5f, dt = %0.3e\r", par_host->time, par_host->dt);
	std::cout.flush();
}

void LoquatSph::PredictorSection()
{
	//========================Neighbor search========================
	timer->clock_neighbor_search = std::clock();
	grid->NeighborSearch(par_host, par_device, cellpos, idmassidx);
	SortArraysPrimary();
	cudaDeviceSynchronize();
	timer->neighbor_search += (std::clock() - timer->clock_neighbor_search) / static_cast<double>(CLOCKS_PER_SEC);

	//=======================Boundary treatment=======================
	timer->clock_adami_boundary = std::clock();
	if (par_host->n_motion > 0)RigidBodyUpdate();
	AdamiBoundary();
	FindCornerParticle();
	//IntegrateParticleMaterial();
	cudaDeviceSynchronize();	
	if (par_host->stress_shepard_filter != 0 && step % par_host->stress_shepard_filter == 0 && step != 0) StressRegularization();
	timer->adami_boundary += (std::clock() - timer->clock_adami_boundary) / static_cast<double>(CLOCKS_PER_SEC);

	//==========================Interaction===========================
	timer->clock_interaction = std::clock();
	ParticleInteraction();
	cudaDeviceSynchronize();
	timer->interaction += (std::clock() - timer->clock_interaction) / static_cast<double>(CLOCKS_PER_SEC);

	//==========================Integration===========================
	timer->clock_integration = std::clock();
	Predictor();
	cudaDeviceSynchronize();
	timer->integration += (std::clock() - timer->clock_integration) / static_cast<double>(CLOCKS_PER_SEC);
}

void LoquatSph::CorrectorSection()
{
	//========================Neighbor search========================
	timer->clock_neighbor_search = std::clock();
	grid->NeighborSearch(par_host, par_device, cellpos, idmassidx);
	SortArraysAll();
	cudaDeviceSynchronize();
	timer->neighbor_search += (std::clock() - timer->clock_neighbor_search) / static_cast<double>(CLOCKS_PER_SEC);

	//=======================Boundary treatment=======================
	timer->clock_adami_boundary = std::clock();
	AdamiBoundary();
	FindCornerParticle();
	//IntegrateParticleMaterial();
	cudaDeviceSynchronize();
	timer->adami_boundary += (std::clock() - timer->clock_adami_boundary) / static_cast<double>(CLOCKS_PER_SEC);

	//==========================Interaction===========================
	timer->clock_interaction = std::clock();
	ParticleInteraction();
	cudaDeviceSynchronize();
	timer->interaction += (std::clock() - timer->clock_interaction) / static_cast<double>(CLOCKS_PER_SEC);

	//==========================Integration===========================
	timer->clock_integration = std::clock();
	Corrector();
	cudaDeviceSynchronize();
	timer->integration += (std::clock() - timer->clock_integration) / static_cast<double>(CLOCKS_PER_SEC);
}

//Outoup controling parameters and particle information 
void LoquatSph::Saving(std::ofstream& time_history_ofs)
{
	timer->clock_saving = std::clock();

	double time_interval = (std::clock() - timer->clock_interval) / static_cast<double>(CLOCKS_PER_SEC);
	double fps = static_cast<double>(step - step_interval + 1) / time_interval;

	double used_time = (std::clock() - timer->clock_initial) / static_cast<double>(CLOCKS_PER_SEC);
	double predicted_time = (par_host->time_max / par_host->time) * used_time;

	printf(" \n");
	printf("----------Save-----------\n");
	printf("Time = %0.5f | Step = %d | FPS = %0.5f\n", par_host->time, step, fps);
	std::cout << "Clock time total : " << TimeToString(predicted_time) << std::endl;
	std::cout << "Clock time used  : " << TimeToString(used_time) << std::endl;
	std::cout << "Clock time needed: " << TimeToString(predicted_time - used_time) << std::endl;
	printf("Number of particles: %d | Number of cells: %d\n", par_host->np, par_host->ncell);
	
	par_host->output_number++;			
	CopyDataFromDeviceToHost();
	io->SaveXml();
	io->SaveParticles();				

	GetValuesAlongLine(project_name, par_host, par_device, pospres, velrhop, idmassidx, contact_mag, contact_var, str, stn, cellpos, grid->GetCellBeg(), grid->GetCellEnd());
	//InterpolateResults(project_name, par_host, par_device, pospres, velrhop, idmassidx, contact_mag, contact_var, str, cellpos, grid->GetCellBeg(), grid->GetCellEnd(), time_history_ofs);
	GetValuesAtPoint(project_name, par_host, par_device, pospres, velrhop, idmassidx, contact_mag, contact_var, str, cellpos, grid->GetCellBeg(), grid->GetCellEnd(), time_history_ofs);

	std::cout << "Save file: " << project_name << "/";
	if(par_host->output_format == 1)
	{
		std::cout << par_host->output_number << ".dat + " << par_host->output_number << ".xml" << std::endl;
	}
	else if(par_host->output_format == 2)
	{
		std::cout << "*_" << par_host->output_number << ".vtu + " << par_host->output_number << ".xml" << std::endl;
	}
	std::cout << std::endl;

	output_time = 0.0;
	step_interval = step;
	timer->clock_interval = std::clock();

	timer->saving += (std::clock() - timer->clock_saving) / static_cast<double>(CLOCKS_PER_SEC);
}

//Summary of the whole calculation
void LoquatSph::SummaryToCalculation()
{
	timer->total = (std::clock() - timer->clock_initial) / static_cast<double>(CLOCKS_PER_SEC);
	double fps_average = static_cast<double>(step) / (timer->total - timer->saving);

	std::cout << "Simulation finished ..." << std::endl;
	std::cout << std::endl;
	std::cout << "----------SUMMARY----------" << std::endl;
	std::cout << "1. Total clock time = " << TimeToString(timer->total) << std::endl;
	std::cout << "2. Search and sorting = " << TimeToString(timer->neighbor_search) << std::endl;
	std::cout << "3. Boundary treatment = " << TimeToString(timer->adami_boundary) << std::endl;
	std::cout << "4. Interpolation = " << TimeToString(timer->interpolation) << std::endl;
	std::cout << "5. Interaction = " << TimeToString(timer->interaction) << std::endl;
	std::cout << "6. Integration = " << TimeToString(timer->integration) << std::endl;
	std::cout << "7. Saving = " << TimeToString(timer->saving) << std::endl;
	std::cout << "8. Total steps = " << step << std::endl;
	std::cout << "9. Average time step = " << par_host->time / static_cast<scalar>(step) << std::endl;
	std::cout << "10. Average FPS = " << fps_average << std::endl;

	std::vector<scalar> max_displacements = io->GetVecMaxDisp();
	for(int i = 0; i < max_displacements.size(); i++)
	{
		printf("%0.3e  \n", max_displacements[i]);
	}
}

void LoquatSph::RunSimulation(int argc, char* argv[])
{
	//Create new output folder and obtain GPU infomation
	GPUInfoExibition(argc, argv);

	//Data input   
    io->LoadParameters(); // Read controling parameters
    io->LoadParticles();  // Read particle information
	par_host = io->GetParameters();

	//Initialization of kernel function and memory allocation 
	InitializeParameters();
	AllocateGpuMemory();
	CopyDataFromHostToDevice();
	checkCudaErrors(cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice));

	//Linked-list generation and initial neighbor search 
	grid->GridConfig(par_host);
	grid->PositionFromAbsolute2Relative(par_host, par_device, cellpos, pospres);
	grid->NeighborSearch(par_host, par_device, cellpos, idmassidx);
	SortArraysPrimary();

	//Output of initial configuration
	printf("Saving initial state ...\n");
	CopyDataFromDeviceToHost();
	io->SaveXml();
	io->SaveParticles();

	//Timer initiation
	TimerInitiation();

	//Subfolder generation for containing specific output
	std::stringstream ss;
	ss << project_name << "/time_history" << ".txt";
	std::string filename = ss.str();
	std::ofstream time_history_ofs(filename.c_str(), std::ofstream::out);

	while(time < static_cast<double>(par_host->time_max)){ //Loop. The main body
		// Calculate step interval and print output header
		StepCalculation();

		// Predictor-Corrector
		PredictorSection();
		CorrectorSection();

		// Saving
		if(output_time > static_cast<double>(par_host->output_frequency)) Saving(time_history_ofs);
		time += static_cast<double>(par_host->dt);
		output_time += static_cast<double>(par_host->dt);
		step++;
	}

	time_history_ofs.close();
	SummaryToCalculation();
}
