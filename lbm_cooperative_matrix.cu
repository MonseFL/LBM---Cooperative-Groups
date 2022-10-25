
#include <string>
#include <math.h>
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/types.h> 
#include <sys/stat.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#if defined(_WIN32)  
#include <direct.h> 
#endif

typedef double prec;

#define BLOCK_SIZE 25;
#define X_SIZE 5;
#define Y_SIZE 5;
#define FILE_BATCH_SIZE 1000000
#define EARTH_VEL 0.000072921

typedef struct mainHStruct {
	int* node_types;
	prec* b;
	prec* w;
	int* node_values;
	int* TSind;
	prec* TSdata;
} mainHStruct;

typedef struct mainDStruct {
	int Lx;
	int Ly;
	int NTS;
	int TTS;
	int Nblocks;
	int Ngrid;
	int Nblocks_real;
	int Ngrid_real;
	int* node_types;
	prec* b;
	prec* w;
	int* node_values;
	int* TSind;
	prec* TSdata;
} mainDStruct;

typedef struct cudaStruct {
	prec tau;
	prec g;
	prec e;
	int* ex;
	int* ey;
	unsigned char* SC_bin;
	unsigned char* BB_bin;

	prec* h;
	prec* ux;
	prec* uy;
	prec* f1;
	prec* f2;
	prec* force;
} cudaStruct;


prec stod(char* word, int len) {
	prec val = 0, ord = 1E-16;
	int dig;
	for (int i = len - 1; i >= 0; i--) {
		if (word[i] != '.') {
			dig = word[i] - '0';
			val += dig * ord;
			ord *= 10;
		}
	}
	return val;
}

void readInput(prec** b, prec** w,
	int** node_types, int** node_values, std::string inputdir,
	int* Lx, int* Ly, prec* Dx, prec* x0, prec* y0, int x_size, int y_size) {
	FILE* fp;
	std::string fullfile = inputdir + ".txt";
	if ((fp = fopen(fullfile.c_str(), "r")) == NULL) {
		std::cout << "Input file doesn't exist." << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "Reading input from " << fullfile << std::endl;

	fscanf(fp, "%d %d %lf %lf %lf\n", Lx, Ly, Dx, x0, y0);

	int Lx_extras = 0;
	int Ly_extras = 0;

	int H_blocks = (int)ceil(((prec)(*Lx) - (prec)x_size) / (x_size - 1)) + 1;
	int V_blocks = (int)ceil(((prec)(*Lx) - (prec)y_size) / (y_size - 1)) + 1;


	if (((*Lx) -x_size) % (x_size -1 ) != 0) {
		int bloquesX = (*Lx) / x_size;
		Lx_extras =(x_size-1)- ((*Lx) - x_size) % (x_size - 1);

	}

	if (((*Ly) - y_size) % (y_size - 1) != 0) {
		int bloquesY = (*Ly) / y_size;
		Ly_extras = (y_size - 1) - ((*Ly) - y_size) % (y_size - 1);

	}


	prec* bl = new prec[((*Lx) + Lx_extras) * ((*Ly)+ Ly_extras)];
	prec* wl = new prec[((*Lx) + Lx_extras) * ((*Ly) + Ly_extras)];
	int* node_typesl = new int[((*Lx) + Lx_extras) * ((*Ly) + Ly_extras)];
	int* node_valuesl = new int[((*Lx) + Lx_extras) * ((*Ly) + Ly_extras)];
	int wc = 0, bc = 0, len = 0, buflen;
	prec val;
	char buffer[FILE_BATCH_SIZE], word[50];
	buffer[FILE_BATCH_SIZE - 1] = '\0';

	
	while (wc < ((*Lx) + Lx_extras) * ((*Ly) + Ly_extras)) {
		buflen = fread(buffer, 1, FILE_BATCH_SIZE - 1, fp);
		for (int i = 0; i <= buflen; i++) {
			if (wc - ((wc / ((*Lx) + Lx_extras)) + 1) * (*Lx) >= 0) {
				wl[wc] = 0;
				node_typesl[wc] = 0;
				wc++;
				bl[bc] = 0;
				bc++;

			}
			else if ((wc / ((*Ly) + Ly_extras)) - (*Ly) >= 0) {
				wl[wc] = 0;
					node_typesl[wc] = 0;
					wc++;
					bl[bc] = 0;
					bc++;
			}
			else {
				if (buffer[i] == ' ' || buffer[i] == '\n' || buffer[i] == '\r') {
					word[len] = '\0';
						if (len == 1) {

							node_typesl[wc - 1] = word[0] - '0';
							//std::cout << node_typesl[wc - 1];
						}
						else if (len > 1) {
							val = stod(word, len);
							if (wc == bc) {
								bl[bc] = val;
								bc++;
							}
							else {
								wl[wc] = val;
								wc++;
							}
						}
					len = 0;
					}
				else {
					word[len] = buffer[i];
					len++;
				}

			}
		}
		if (buflen != FILE_BATCH_SIZE - 1)
			break;
		if (buffer[buflen - 1] != ' ' && buffer[buflen - 1] != '\n' && buffer[buflen - 1] != '\r')
			fseek(fp, 1 - len, SEEK_CUR);
		len = 0;
	}
	fclose(fp);
	*w = wl;
	*b = bl;
	*node_types = node_typesl;

	int contador_x = 1;
	int contador_y = 1;
	int borde_x, borde_y;
	int y, x;
	for (int i = 0; i < ((*Lx) + Lx_extras) * ((*Ly) + Ly_extras); i++) {
		y = i / ((*Lx) + Lx_extras);
		x = i % ((*Ly) + Ly_extras);
		if (x == ((*Lx) + Lx_extras) - 1) {
			contador_x = 1;

		}


		borde_x = contador_x * x_size - contador_x;
		borde_y = contador_y * y_size - contador_y;
		if (x == borde_x && y == borde_y) {
			node_valuesl[i] = 4;

			contador_x++;
		}
		else if (x == borde_x || y == borde_y) {
			node_valuesl[i] = 2;
			contador_x = (x == borde_x) ? contador_x + 1 : contador_x;
		}
		else {
			node_valuesl[i] = 1;
		}
		if (x == ((*Lx) + Lx_extras) - 1 && y == borde_y && y != ((*Ly) + Ly_extras) - y_size) {
			contador_y++;
		}

	}


	*node_values = node_valuesl;
}

//void readInput(prec** b, prec** w,
//	int** node_types,int** node_values,
//	int* Lx, int* Ly, prec* Dx, prec* x0, prec* y0, int x_size, int y_size) {
//	FILE* fp;
//	std::string fullfile = "test_63x64.txt";
//	if ((fp = fopen(fullfile.c_str(), "r")) == NULL) {
//		std::cout << "Input file doesn't exist." << std::endl;
//		exit(EXIT_FAILURE);
//	}
//	std::cout << "Reading input from " << fullfile << std::endl;
//
//	fscanf(fp, "%d %d %lf %lf %lf\n", Lx, Ly, Dx, x0, y0);
//
//	std::cout << "datos leidos\n";
//
//	prec* bl = new prec[(*Lx) * (*Ly)];
//	prec* wl = new prec[(*Lx) * (*Ly)];
//	int* node_typesl = new int[(*Lx) * (*Ly)];
//	int* node_valuesl = new int[(*Lx) * (*Ly)];
//	//int wc = 0, bc = 0, len = 0, buflen;
//	prec val;
//	int y, x;
//	int contador_x = 1;
//	int contador_y = 1;
//	int borde_x, borde_y;
//	for (int i = 0; i < (*Lx) * (*Ly); i++) {
//
//		fscanf(fp, "%lf %lf", &bl[i], &wl[i]);
//
//		y = i / (*Lx);
//		x = i % (*Lx);
//		if (x == (*Lx) - 1) {
//			node_typesl[i] = 2;
//		}
//		else if (x == 0 || y == 0 || y == (*Ly) - 1) {
//			node_typesl[i] = 2;
//		}
//		else {
//			node_typesl[i] = 0;
//		}
//
//		if (x == (*Lx) - 1) {
//			contador_x = 1;
//
//		}
//
//		
//		borde_x = contador_x * x_size - contador_x;
//		borde_y = contador_y * y_size - contador_y;
//		if (x ==borde_x && y == borde_y ) {
//			node_valuesl[i] = 4;
//
//			contador_x++;
//		}
//		else if (x == borde_x || y == borde_y) {
//			node_valuesl[i] = 2;
//			contador_x = (x == borde_x) ? contador_x + 1 : contador_x;
//		}
//		else {
//			node_valuesl[i] = 1;
//		}
//		if (x == (*Lx) - 1 && y == borde_y && y != (*Ly) - y_size) {
//			contador_y++;
//		}
//		
//	}
//	std::cout << "arreglo iniciado";
//	fclose(fp);
//	*w = wl;
//	*b = bl;
//	*node_types = node_typesl;
//	*node_values = node_valuesl;
//}







void freemem(mainHStruct host, mainDStruct devi, cudaStruct devEx) {
	/*delete[] host.b;
	delete[] host.w;
	delete[] host.ux;
	delete[] host.uy;
	delete[] host.node_types;
	*/

	cudaFree(devi.b);
	cudaFree(devi.w);
	cudaFree(devi.node_types);
	cudaFree(devi.TSind);
	cudaFree(devi.TSdata);

	cudaFree(devEx.ex);
	cudaFree(devEx.ey);
	cudaFree(devEx.h);
	cudaFree(devEx.f1);
	cudaFree(devEx.f2);
#if IN == 3
	cudaFree(devEx.Arr_tri);
#elif IN == 4
	cudaFree(devEx.SC_bin);
	cudaFree(devEx.BB_bin);
#endif
}






__global__ void LBMpush(int Lx, int Ly, prec g, prec e, prec tau, const unsigned char* __restrict__ SC_bin,
	const unsigned char* __restrict__ BB_bin, int* node_value, const prec* __restrict__ f1,
	prec* f2, prec* h, prec* b, prec* ux, prec* uy, int x_size, int y_size, int block_size, int H_blocks, int t) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int size = Lx * Ly;
	thread_block block_group = this_thread_block();


	int neigh[8];
	prec ftemp[9], ftemp2[9];
	prec source[8];
	prec feq[9];

	prec factor = 1 / (6 * e * e);
	prec factor21 = -(1 * g / 3.0) / (e * e);
	prec factor22 = -(1 * g / 12.0) / (e * e);
	prec tauVal = tau - 1;


	prec ux3, uy3, uxuy5, uxuy6;
	prec fact1 = 1 / (9 * e * e);
	prec fact2 = fact1 * 0.25;

	prec gh, usq;
	unsigned char SC, BB;



	// calculo de indices
	int x_local = i % x_size;
	int y_local = i % block_size / x_size;

	int y = y_local + (blockIdx.x / H_blocks) * y_size - (blockIdx.x / H_blocks);
	int x = x_local + (blockIdx.x % H_blocks) * x_size - (blockIdx.x % H_blocks);

	int i_read = x + y * Lx;

	SC = SC_bin[i_read];
	BB = BB_bin[i_read];

	if (SC + BB != 0) {

		//calculo vecinos
		neigh[0] = (x != Lx - 1) ? i_read + 1 : i_read;
		neigh[1] = (y != 0) ? i_read - Lx : i_read;
		neigh[2] = (x != 0) ? i_read - 1 : i_read;
		neigh[3] = (y != Ly - 1) ? i_read + Lx : i_read;
		neigh[4] = (y != 0 && x != Lx - 1) ? i_read - Lx + 1 : i_read;
		neigh[5] = (y != 0 && x != 0) ? i_read - Lx - 1 : i_read;
		neigh[6] = (y != Ly - 1 && x != 0) ? i_read + Lx - 1 : i_read;
		neigh[7] = (y != Ly - 1 && x != Lx - 1) ? i_read + Lx + 1 : i_read;

		//Variables locales

		prec localh = h[i_read];
		prec localux = ux[i_read];
		prec localuy = uy[i_read];




		//Source term
		source[0] = factor21 * 0.5 * (h[neigh[0]] + localh) * (b[neigh[0]] - b[i_read]);
		source[1] = factor21 * 0.5 * (h[neigh[1]] + localh) * (b[neigh[1]] - b[i_read]);
		source[2] = factor21 * 0.5 * (h[neigh[2]] + localh) * (b[neigh[2]] - b[i_read]);
		source[3] = factor21 * 0.5 * (h[neigh[3]] + localh) * (b[neigh[3]] - b[i_read]);
		source[4] = factor22 * 0.5 * (h[neigh[4]] + localh) * (b[neigh[4]] - b[i_read]);
		source[5] = factor22 * 0.5 * (h[neigh[5]] + localh) * (b[neigh[5]] - b[i_read]);
		source[6] = factor22 * 0.5 * (h[neigh[6]] + localh) * (b[neigh[6]] - b[i_read]);
		source[7] = factor22 * 0.5 * (h[neigh[7]] + localh) * (b[neigh[7]] - b[i_read]);





		gh = 1.5 * g * localh;
		usq = 1.5 * (localux * localux + localuy * localuy);
		ux3 = 3.0 * e * localux;
		uy3 = 3.0 * e * localuy;
		uxuy5 = ux3 + uy3;
		uxuy6 = uy3 - ux3;

		feq[0] = localh - fact1 * localh * (5.0 * gh + 4.0 * usq);
		feq[1] = fact1 * localh * (gh + ux3 + 0.5 * ux3 * ux3 * 9 * fact1 - usq);
		feq[2] = fact1 * localh * (gh + uy3 + 0.5 * uy3 * uy3 * 9 * fact1 - usq);
		feq[3] = fact1 * localh * (gh - ux3 + 0.5 * ux3 * ux3 * 9 * fact1 - usq);
		feq[4] = fact1 * localh * (gh - uy3 + 0.5 * uy3 * uy3 * 9 * fact1 - usq);
		feq[5] = fact2 * localh * (gh + uxuy5 + 0.5 * uxuy5 * uxuy5 * 9 * fact1 - usq);
		feq[6] = fact2 * localh * (gh + uxuy6 + 0.5 * uxuy6 * uxuy6 * 9 * fact1 - usq);
		feq[7] = fact2 * localh * (gh - uxuy5 + 0.5 * uxuy5 * uxuy5 * 9 * fact1 - usq);
		feq[8] = fact2 * localh * (gh - uxuy6 + 0.5 * uxuy6 * uxuy6 * 9 * fact1 - usq);


		//Streaming
		f2[i_read] = (f1[i_read] * tauVal + feq[0]) / tau;
		if (neigh[0] != i_read) f2[neigh[0] + size] = (f1[i_read + size] * tauVal + feq[1]) / tau + source[0];
		if (neigh[1] != i_read) f2[neigh[1] + 2 * size] = (f1[i_read + 2 * size] * tauVal + feq[2]) / tau + source[1];
		if (neigh[2] != i_read) f2[neigh[2] + 3 * size] = (f1[i_read + 3 * size] * tauVal + feq[3]) / tau + source[2];
		if (neigh[3] != i_read) f2[neigh[3] + 4 * size] = (f1[i_read + 4 * size] * tauVal + feq[4]) / tau + source[3];
		if (neigh[4] != i_read) f2[neigh[4] + 5 * size] = (f1[i_read + 5 * size] * tauVal + feq[5]) / tau + source[4];
		if (neigh[5] != i_read) f2[neigh[5] + 6 * size] = (f1[i_read + 6 * size] * tauVal + feq[6]) / tau + source[5];
		if (neigh[6] != i_read) f2[neigh[6] + 7 * size] = (f1[i_read + 7 * size] * tauVal + feq[7]) / tau + source[6];
		if (neigh[7] != i_read) f2[neigh[7] + 8 * size] = (f1[i_read + 8 * size] * tauVal + feq[8]) / tau + source[7];



		block_group.sync();


		int old = atomicAdd(&node_value[i_read], 1);
		int type = old / t;
		old = (type < 4) ? old % 2 : 4 - old % 4;
		if (old == 1 || type == 1) {



			ftemp2[0] = f2[i_read];
			ftemp2[1] = f2[i_read + size];
			ftemp2[2] = f2[i_read + 2 * size];
			ftemp2[3] = f2[i_read + 3 * size];
			ftemp2[4] = f2[i_read + 4 * size];
			ftemp2[5] = f2[i_read + 5 * size];
			ftemp2[6] = f2[i_read + 6 * size];
			ftemp2[7] = f2[i_read + 7 * size];
			ftemp2[8] = f2[i_read + 8 * size];




			if ((SC >> 0) & 1) ftemp2[1] = (f1[i_read + size] * tauVal + feq[1]) / tau;
			if ((SC >> 1) & 1) ftemp2[2] = (f1[i_read + 2 * size] * tauVal + feq[2]) / tau;
			if ((SC >> 2) & 1) ftemp2[3] = (f1[i_read + 3 * size] * tauVal + feq[3]) / tau;
			if ((SC >> 3) & 1) ftemp2[4] = (f1[i_read + 4 * size] * tauVal + feq[4]) / tau;
			if ((SC >> 4) & 1) ftemp2[5] = (f1[i_read + 5 * size] * tauVal + feq[5]) / tau;
			if ((SC >> 5) & 1) ftemp2[6] = (f1[i_read + 6 * size] * tauVal + feq[6]) / tau;
			if ((SC >> 6) & 1) ftemp2[7] = (f1[i_read + 7 * size] * tauVal + feq[7]) / tau;
			if ((SC >> 7) & 1) ftemp2[8] = (f1[i_read + 8 * size] * tauVal + feq[8]) / tau;

			ftemp2[1] = ((BB >> (0)) & 1) ? f1[i_read + 3 * size] : ftemp2[1];
			ftemp2[2] = ((BB >> (1)) & 1) ? f1[i_read + 4 * size] : ftemp2[2];
			ftemp2[3] = ((BB >> (2)) & 1) ? f1[i_read + 1 * size] : ftemp2[3];
			ftemp2[4] = ((BB >> (3)) & 1) ? f1[i_read + 2 * size] : ftemp2[4];
			ftemp2[5] = ((BB >> (4)) & 1) ? f1[i_read + 7 * size] : ftemp2[5];
			ftemp2[6] = ((BB >> (5)) & 1) ? f1[i_read + 8 * size] : ftemp2[6];
			ftemp2[7] = ((BB >> (6)) & 1) ? f1[i_read + 5 * size] : ftemp2[7];
			ftemp2[8] = ((BB >> (7)) & 1) ? f1[i_read + 6 * size] : ftemp2[8];



			f2[i_read + size] = ftemp2[1];
			f2[i_read + 2 * size] = ftemp2[2];
			f2[i_read + 3 * size] = ftemp2[3];
			f2[i_read + 4 * size] = ftemp2[4];
			f2[i_read + 5 * size] = ftemp2[5];
			f2[i_read + 6 * size] = ftemp2[6];
			f2[i_read + 7 * size] = ftemp2[7];
			f2[i_read + 8 * size] = ftemp2[8];

			h[i_read] = ftemp2[0] + (ftemp2[1] + ftemp2[2] + ftemp2[3] + ftemp2[4]) + (ftemp2[5] + ftemp2[6] + ftemp2[7] + ftemp2[8]);
			ux[i_read] = e * ((ftemp2[1] - ftemp2[3]) + (ftemp2[5] - ftemp2[6] - ftemp2[7] + ftemp2[8])) / h[i_read];
			uy[i_read] = e * ((ftemp2[2] - ftemp2[4]) + (ftemp2[5] + ftemp2[6] - ftemp2[7] - ftemp2[8])) / h[i_read];

			//calculo variables macroscopicas




		}

	}
}




//__global__ void auxArraysKernel(int Lx, int Ly,
//	const int* __restrict__ ex, const int* __restrict__ ey,
//	const int* __restrict__ node_types,
//	unsigned char* SC_bin, unsigned char* BB_bin) {
//
//
//
//	int i = threadIdx.x + blockIdx.x * blockDim.x;
//
//
//	int size = Lx * Ly;
//	if (i < size) {
//		int y = (int)i / Lx;
//		int x = i - y * Lx;
//		//int xi, yi, ind, indj, indk, a;
//		int valueSC = 0, valueBB = 0;
//		if (node_types[i] == 1) {
//			valueBB = 64 + 32 + 8;
//			if (y == 0) {
//				valueSC = 8 + 128;
//			}
//			if (y == Ly - 1) {
//				valueSC = 2 + 16;
//			}
//		}
//		else if (node_types[i] == 2) {
//			if (y == 0) {
//				valueSC = 8 + 64 + 128;
//				if (x == 0) {
//					valueSC = 1 + 8 + 16 + 64 + 128;
//				}
//				if (x == Lx - 1) {
//					valueSC = 4 + 8 + 32 + 64 + 128;
//				}
//			}
//			else if (y == Ly - 1) {
//				valueSC = 2 + 16 + 32;
//				if (x == 0) {
//					valueSC = 1 + 2 + 16 + 32 + 128;
//				}
//				if (x == Lx - 1) {
//					valueSC = 2 + 4 + 16 + 32 + 64;
//				}
//
//			}
//			else if (x == 0) {
//				valueSC = 128 + 16 + 1;
//			}
//			else if (x == Lx - 1) {
//				valueSC = 4 + 32 + 64;
//			}
//
//		}
//
//		SC_bin[i] = (unsigned char)valueSC;
//		BB_bin[i] = (unsigned char)valueBB;
//	}
//}

__global__ void auxArraysKernel(int Lx, int Ly, int Lxo, int Lyo,
	const int* __restrict__ ex, const int* __restrict__ ey,
	const int* __restrict__ node_types,
	unsigned char* SC_bin, unsigned char* BB_bin) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int size = Lx * Ly;
	if (i < size) {
		int y = (int)i / Lx;
		int x = i - y * Lx;
		int xi, yi, ind, indj, indk, a;
		int valueSC = 0, valueBB = 0;
		if (node_types[i] == 2) {
			if (y == 0) {
				if (x == 0)
					valueSC += 1 + 8 + 16 + 64 + 128;
				else if (x == Lxo - 1)
					valueSC += 4 + 8 + 32 + 64 + 128;
				else
					valueSC += 8 + 64 + 128;
			}
			else if (y == Lyo - 1) {
				if (x == 0)
					valueSC += 1 + 2 + 16 + 32 + 128;
				else if (x == Lxo - 1)
					valueSC += 2 + 4 + 16 + 32 + 64;
				else
					valueSC += 2 + 16 + 32;
			}
			else {
				if (x == 0)
					valueSC += 1 + 16 + 128;
				else if (x == Lxo - 1)
					valueSC += 4 + 32 + 64;
				else
					valueSC = 0;
			}
		}
		else if (node_types[i] == 1) {
			if (y == 0) {
				if (x == 0) {
					valueBB += 1 + 8 + 128;
					valueSC += 16 + 64;
				}
				else if (x == Lxo - 1) {
					valueBB += 4 + 8 + 64;
					valueSC += 32 + 128;
				}
				else {
					valueBB += 8 + 64 + 128;
				}
			}
			else if (y == Lyo - 1) {
				if (x == 0) {
					valueBB += 1 + 2 + 16;
					valueSC += 32 + 128;
				}
				else if (x == Lxo - 1) {
					valueBB += 2 + 4 + 32;
					valueSC += 16 + 64;
				}
				else {
					valueBB += 2 + 16 + 32;
				}

			}
			else {
				if (x == 0)
					valueBB += 1 + 16 + 128;
				else if (x == Lx - 1)
					valueBB += 4 + 32 + 64;
				else
					valueBB = 0;
			}

		}
		else {
			valueBB += 0;
			valueSC += 0;
		}
		SC_bin[i] = (unsigned char)valueSC;
		BB_bin[i] = (unsigned char)valueBB;
	}
}

__global__ void hKernel(int Lx, int Ly, const prec* __restrict__ w,
	const prec* __restrict__ b, prec* h) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < Lx * Ly) {
		h[i] = w[i] - b[i];
	}
}

__global__ void feqKernel(int Lx, int Ly, prec g, prec e,
	const prec* __restrict__ h, prec* f) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < Lx * Ly) {
		prec hi = h[i];
		prec gh1 = g * hi * hi / (6.0 * e * e);
		prec gh2 = gh1 / 4;
		//printf("%d %f %f %f %f\n", i, gh1, hi, g, e);
		f[i] = hi - 5.0 * gh1;
		//printf("%f\n", f[i]);
		f[i + (Lx * Ly)] = gh1;
		f[i + (2 * Lx * Ly)] = gh1;
		f[i + (3 * Lx * Ly)] = gh1;
		f[i + (4 * Lx * Ly)] = gh1;
		f[i + (5 * Lx * Ly)] = gh2;
		f[i + (6 * Lx * Ly)] = gh2;
		f[i + (7 * Lx * Ly)] = gh2;
		f[i + (8 * Lx * Ly)] = gh2;
	}
}

void setup(mainDStruct devi, cudaStruct devEx, int x_size, int y_size, int H_blocks, int Lxo, int Lyo) {

	cudaError_t err = cudaGetLastError();
	printf("\n aux init\n");
	auxArraysKernel << <devi.Ngrid_real, devi.Nblocks >> > (devi.Lx, devi.Ly,Lxo,Lyo, devEx.ex, devEx.ey, devi.node_types,
		devEx.SC_bin, devEx.BB_bin);
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
	printf("\nauxArraysKernel finish\n");

	hKernel << <devi.Ngrid_real, devi.Nblocks >> > (devi.Lx, devi.Ly, devi.w, devi.b, devEx.h);
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
	printf("\n hKernel finish\n");
	printf("\n feqKernel init\n");

	feqKernel << <devi.Ngrid_real, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.h, devEx.f1);

	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}

	printf("\n feqKernel finish\n");

	//TSkernel << <devi.NTS, 1 >> > (devi.TSdata, devi.w, devi.TSind, 0, deltaTS, devi.NTS, devi.TTS);
}


__global__ void wKernel(int Lx, int Ly, const prec* __restrict__ h,
	const prec* __restrict__ b, prec* w) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < Lx * Ly) {
		w[i] = h[i] + b[i];
	}
}

__global__ void orderKernel(int Lx, int Ly, int H_blocks, int y_size, int x_size, const prec* __restrict__ h,
	const prec* __restrict__ b, prec* w) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	int x_local = i % x_size;
	int y_local = i / x_size - blockIdx.x * y_size;


	int y = y_local + (blockIdx.x / H_blocks) * y_size;
	int x = x_local + (blockIdx.x % H_blocks) * x_size;

	int ind = x + y * x_size;

	if (i < Lx * Ly) {
		w[i] = h[ind] + b[ind];
	}


}

void writeOutput(int L, int t, prec* w, std::string outputdir) {
	FILE* fp;
	std::ostringstream numero;
	numero << std::setw(5) << std::setfill('0') << std::right << (t);
	std::string fullfile = outputdir + ".dat";
	if ((fp = fopen(fullfile.c_str(), "wb")) == NULL) {
		std::cout << "Can't create output file." << std::endl;
		exit(EXIT_FAILURE);
	}
	fwrite(&w[0], sizeof(prec), L, fp);
	fclose(fp);
}
void writeOutput2(int L, int t, int* w, std::string outputdir) {
	FILE* fp;
	std::ostringstream numero;
	numero << std::setw(5) << std::setfill('0') << std::right << (t);
	std::string fullfile = outputdir + ".dat";
	if ((fp = fopen(fullfile.c_str(), "wb")) == NULL) {
		std::cout << "Can't create output file." << std::endl;
		exit(EXIT_FAILURE);
	}
	fwrite(&w[0], sizeof(int), L, fp);
	fclose(fp);
}
void copyAndWriteResultData(mainHStruct host, mainDStruct devi, cudaStruct devEx, int t, std::string outputdir) {

	wKernel << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.h, devi.b, devi.w);


	cudaMemcpy(host.w, devi.w, devi.Lx * devi.Ly * sizeof(prec), cudaMemcpyDeviceToHost);

	writeOutput(devi.Lx * devi.Ly, t, host.w, outputdir);

}

void copyAndWriteResultData_v2(mainHStruct host, mainDStruct devi, cudaStruct devEx, int t, std::string outputdir) {

	wKernel << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.h, devi.b, devi.w);


	cudaMemcpy(host.w, devi.w, devi.Lx * devi.Ly * sizeof(prec), cudaMemcpyDeviceToHost);

	writeOutput(devi.Lx * devi.Ly, t, host.w, outputdir);

}




int main() {

	int Lx, Ly, Nblocks;
	prec Dx, x0, y0, tau, g, Dt;


	int x_size, iter;



	std::string inputdir;
	std::string outputdir = "output_SB_";

	std::cout << "Test file : ";
	std::cin >> inputdir;
	std::cout << "Block size : ";
	std::cin >> Nblocks;
	std::cout << "Group width : ";
	std::cin >> x_size;
	std::cout << "Dt : ";
	std::cin >> Dt;
	std::cout << "tau : ";
	std::cin >> tau;
	std::cout << "g : ";
	std::cin >> g;
	std::cout << "iter: ";
	std::cin >> iter;


	outputdir = outputdir + inputdir;

	mainHStruct host;
	mainDStruct devi;
	cudaStruct devEx;

	int block_size = Nblocks;
	int y_size = Nblocks / x_size;



	readInput(&host.b, &host.w, &host.node_types, &host.node_values, inputdir, &Lx, &Ly, &Dx, &x0, &y0, x_size, y_size);


	//writeOutput2(Lx * Ly, 1, host.node_values, "node_values");
	int Lxo = Lx;
	int Lyo = Ly;

	if (((Lx) - x_size) % (x_size - 1) != 0) {
		int bloquesX = (Lx) / x_size;
		int Lx_extras = (x_size - 1) - ((Lx) - x_size) % (x_size - 1);
		Lx = Lx + Lx_extras;

	}

	if (((Ly) - y_size) % (y_size - 1) != 0) {
		int bloquesY = (Ly) / y_size;
		int Ly_extras = (y_size - 1) - ((Ly) - y_size) % (y_size - 1);
		Ly = Ly + Ly_extras;

	}
	int H_blocks = (int)ceil(((prec)Lx - (prec)x_size) / (x_size - 1)) + 1;
	int V_blocks = (int)ceil(((prec)Ly - (prec)y_size) / (y_size - 1)) + 1;

	int inter_x = H_blocks - 1;
	int inter_y = V_blocks - 1;

	int s = Ly * Lx;


	std::cout << "input leido\n";

	double number;


	int num_bytes_d = Lx * Ly * sizeof(prec);

	int num_bytes_i = Lx * Ly * sizeof(int);

	int Ngrid = int(ceil((prec)(Lx + inter_x) * (prec)(Ly + inter_y) / (prec)Nblocks));
	int Ngrid_real = int(ceil((prec)Lx * (prec)Ly / (prec)Nblocks));
	int ex[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
	int ey[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
	prec e = Dx / Dt;

	devi.Lx = Lx;
	devi.Ly = Ly;
	devi.Nblocks = Nblocks;
	devi.Ngrid = Ngrid;

	devi.Ngrid_real = Ngrid;




	std::cout << devi.Lx << "\n";
	std::cout << devi.Ly << "\n";
	std::cout << devi.Nblocks << "\n";
	std::cout << devi.Ngrid << "\n";

	cudaMalloc((void**)&devi.w, num_bytes_d);
	cudaMalloc((void**)&devi.b, num_bytes_d);
	cudaMalloc((void**)&devi.node_types, num_bytes_i);
	cudaMalloc((void**)&devi.node_values, num_bytes_i);



	cudaMemcpy(devi.b, host.b, num_bytes_d, cudaMemcpyHostToDevice);
	cudaMemcpy(devi.w, host.w, num_bytes_d, cudaMemcpyHostToDevice);
	cudaMemcpy(devi.node_types, host.node_types, num_bytes_i, cudaMemcpyHostToDevice);
	cudaMemcpy(devi.node_values, host.node_values, num_bytes_i, cudaMemcpyHostToDevice);


	// cudaMemcpy(devi.TSind, host.TSind, NTS * sizeof(int), cudaMemcpyHostToDevice);

	devEx.tau = tau;
	devEx.g = g;
	devEx.e = e;
	cudaMalloc((void**)&devEx.ex, 9 * sizeof(int));
	cudaMalloc((void**)&devEx.ey, 9 * sizeof(int));
	cudaMalloc((void**)&devEx.h, num_bytes_d);
	cudaMalloc((void**)&devEx.f1, 9 * num_bytes_d);
	cudaMalloc((void**)&devEx.f2, 9 * num_bytes_d);
	cudaMalloc((void**)&devEx.ux, num_bytes_d);
	cudaMalloc((void**)&devEx.uy, num_bytes_d);




	// #if IN == 3
	// 	cudaMalloc((void**)&devEx.Arr_tri, 9 * Lx * Ly * sizeof(unsigned char));
	// #elif IN == 4
	cudaMalloc((void**)&devEx.SC_bin, Lx * Ly * sizeof(unsigned char));
	cudaMalloc((void**)&devEx.BB_bin, Lx * Ly * sizeof(unsigned char));


	// #endif




	int nbloquesDev;

	cudaMemcpy(devEx.ex, ex, 9 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devEx.ey, ey, 9 * sizeof(int), cudaMemcpyHostToDevice);



	clock_t t1, t2;

	size_t a = 0;

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbloquesDev, hKernel, 256, a);


	//std::cout << "resultado:" << nbloquesDev << std::endl;



	int t = 1; //iteraciones
	cudaEvent_t ct1, ct2;
	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);

	float msecs = 0;
	cudaThreadSynchronize();
	setup(devi, devEx, x_size, y_size, H_blocks, Lxo, Lyo);
	std::cout << std::fixed << std::setprecision(1);
	cudaEventRecord(ct1);




	std::string output_file;
	for (int i = 1; i < iter; i++) {

		if (i % 2 != 0) {
			LBMpush << < devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.tau, devEx.SC_bin, devEx.BB_bin, devi.node_values, devEx.f1, devEx.f2, devEx.h, devi.b, devEx.ux, devEx.uy, x_size, y_size, block_size, H_blocks, i);

		}
		else {
			LBMpush << < devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.tau, devEx.SC_bin, devEx.BB_bin, devi.node_values, devEx.f2, devEx.f1, devEx.h, devi.b, devEx.ux, devEx.uy, x_size, y_size, block_size, H_blocks, i);

		}
		cudaError_t err = cudaGetLastError();       // Get error code

		if (err != cudaSuccess)
		{
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
			exit(-1);
		}


		/*cudaMemcpy(devi.node_values, host.node_values, num_bytes_i, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		if ((i - 1) % 500 == 0) {
			cudaMemcpy(host.w, devEx.h, devi.Lx * devi.Ly * sizeof(prec), cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
			output_file = outputdir + "_" + std::to_string(i - 1);
			writeOutput(Lx * Ly, iter, host.w, output_file);
			 for (int i = 0; i < ((Lx)) * ((Ly)); i++) {
			 	if (i % ((Lx)) == 0) {
					std::cout << "\n";
				}
			 	printf("%f ", host.w[i]);

			 }
		}*/


	}

	std::cout << " resultado\n";




	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&msecs, ct1, ct2);
	std::cout << std::endl << "Tiempo total: " << msecs << "[ms]" << std::endl;
	std::cout << std::endl << "Tiempo promedio por iteracion: " << msecs / iter << "[ms]" << std::endl;

	int sharedBytes = 0;

	cudaFree(devi.b);
	cudaFree(devi.w);
	cudaFree(devi.node_types);
	cudaFree(devi.node_values);
	cudaFree(devi.TSind);
	cudaFree(devi.TSdata);

	cudaFree(devEx.ex);
	cudaFree(devEx.ey);
	cudaFree(devEx.h);
	cudaFree(devEx.f1);
	cudaFree(devEx.f2);
	cudaFree(devEx.SC_bin);
	cudaFree(devEx.BB_bin);









	return 0;
}