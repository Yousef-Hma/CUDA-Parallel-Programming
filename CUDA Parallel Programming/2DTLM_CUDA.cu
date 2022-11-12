
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include<fstream>
#include <ctime>

#define M_PI 3.14276 // Value of PI
#define c 299792458 // Speed of light in Vacuum
#define mu0 M_PI*4e-7 // Magnetic permeability in a vacuum
#define eta0 c*mu0  // Wave impedance
#define BLOCKSIZE 1024 // Max threads per block

double** declare_array2D(int, int);

// Function declaring and initaiting a 2D array
double** declare_array2D(int NX, int NY) {
    double** V = new double* [NX];
    for (int x = 0; x < NX; x++) {
        V[x] = new double[NY];
    }

    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            V[x][y] = 0;
        }
    }
    return V;
}

using namespace std;

// GPU kernel function performing the Source Operation on threads across multiple blocks
__global__ void tlmSource(double** V1, double** V2, double** V3, double** V4, double time, int n)
{
    // Variable declarations imported from main()
    double width = 20 * time * sqrt(2.);
    double delay = 100 * time * sqrt(2.);
    int Ein[] = { 10,10 }; // Input

    //Calculate value of gaussian voltage at time point
    double source;
    source = (1 / sqrt(2.)) * exp(-(n * time - delay) * (n * time - delay) / (width * width));
    V1[Ein[0]][Ein[1]] = V1[Ein[0]][Ein[1]] + source;
    V2[Ein[0]][Ein[1]] = V2[Ein[0]][Ein[1]] - source;
    V3[Ein[0]][Ein[1]] = V3[Ein[0]][Ein[1]] - source;
    V4[Ein[0]][Ein[1]] = V4[Ein[0]][Ein[1]] + source;
}

// GPU kernel function performing the Scatter Operation on threads across multiple blocks
__global__ void tlmScatter(double** V1, double** V2, double** V3, double** V4, int NX, int NY)
{
    // Variable declarations imported from main()
    double Z = eta0 / sqrt(2.);
    double I = 0;

    // Variable storing the current thread in the current block ID for each dimension
    unsigned int idx = threadIdx.x + blockDim.x + blockIdx.x;
    unsigned int idy = threadIdx.y + blockDim.y + blockIdx.y;

    double V;

    // Start of scatter function
    if (idx < NX) {
        if (idy < NY) {
            I = (2 * V1[idx][idy] + 2 * V4[idx][idy] - 2 * V2[idx][idy] - 2 * V3[idx][idy]) / (4 * Z);

            V = 2 * V1[idx][idy] - I * Z;         //port1
            V1[idx][idy] = V - V1[idx][idy];
            V = 2 * V2[idx][idy] + I * Z;         //port2
            V2[idx][idy] = V - V2[idx][idy];
            V = 2 * V3[idx][idy] + I * Z;         //port3
            V3[idx][idy] = V - V3[idx][idy];
            V = 2 * V4[idx][idy] - I * Z;         //port4
            V4[idx][idy] = V - V4[idx][idy];
        }
    }
}

// GPU kernel function performing the Connect and Boundary Operations on threads across multiple blocks
__global__ void tlmConnect(double** V1, double** V2, double** V3, double** V4, int NX, int NY)
{
    // Variable declarations imported from main()
    double tempV = 0;
    //boundary coefficients
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    // Variable storing the current thread in the current block ID for each dimension
    unsigned int idx = threadIdx.x + blockDim.x + blockIdx.x;
    unsigned int idy = threadIdx.y + blockDim.y + blockIdx.y;

    // Temporary variables to allow the merging of the Connect and Boundary Operations
    int tempNX, tempNY;
    tempNX = NX - 1;
    tempNY = NY - 1;

    // Start of Connect Operation
    if (idx < NX) {
        if (idy < NY) {
            tempV = V2[idx][idy];
            V2[idx][idy] = V4[idx - 1][idy];
            V4[idx - 1][idy] = tempV;
        }
    }
    if (idx < NX) {
        if (idy < NY) {
            tempV = V1[idx][idy];
            V1[idx][idy] = V3[idx][idy - 1];
            V3[idx][idy - 1] = tempV;

            V4[tempNX][idy] = rXmax * V4[tempNX][idy];  // Start of boundary operation
            V2[0][idy] = rXmin * V2[0][idy];            //....
        }
        V3[idx][tempNY] = rYmax * V3[idx][tempNY];      //....
        V1[idx][0] = rYmin * V1[idx][0];                // End of boundary operation
    }
}


int main() {

    std::clock_t start = std::clock();
    int NX = 100; // Nodes in the X dimension
    int NY = 100; // Nodes in the Y dimension
    int NT = 850000; // Number of Time Steps

    double dl = 1;  // Length of nodes
    double dt = dl / (sqrt(2.) * c); // Set time step duration

    //2D mesh variables
    double** V1 = declare_array2D(NX, NY);
    double** V2 = declare_array2D(NX, NY);
    double** V3 = declare_array2D(NX, NY);
    double** V4 = declare_array2D(NX, NY);

    // Output
    int Eout[] = { 15,15 };

    //device arrays
    double** dev_V1 = declare_array2D(NX, NY);
    double** dev_V2 = declare_array2D(NX, NY);
    double** dev_V3 = declare_array2D(NX, NY);
    double** dev_V4 = declare_array2D(NX, NY);

    //allocate memory on device
    cudaMalloc((void**)&dev_V1, NX * NY * sizeof(double));
    cudaMalloc((void**)&dev_V2, NX * NY * sizeof(double));
    cudaMalloc((void**)&dev_V3, NX * NY * sizeof(double));
    cudaMalloc((void**)&dev_V4, NX * NY * sizeof(double));

    //copy memory from host to device
    cudaMemcpy(dev_V1, V1, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V2, V2, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V3, V3, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V4, V4, NX * NY * sizeof(double), cudaMemcpyHostToDevice);

    ofstream output("output.out");

    // Determination of number of blocks required for the given number of time steps
    // Taking into account the maximum number of threads per block
    int blockSize = BLOCKSIZE;
    int numBlocks = (NT + blockSize - 1) / blockSize;

    for (int n = 0; n < NT; n++) {

        //source
        tlmSource << <numBlocks, blockSize >> > (dev_V1, dev_V2, dev_V3, dev_V4, dt, n);

        //shatter
        tlmScatter << <numBlocks, blockSize >> > (dev_V1, dev_V2, dev_V3, dev_V4, NX, NY);

        //connect
        tlmConnect << <numBlocks, blockSize >> > (dev_V1, dev_V2, dev_V3, dev_V4, NX, NY);

    }

    // Copy memory back from device to host
    cudaMemcpy(V2, dev_V2, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V4, dev_V4, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);

    // Write output to text file
    for (int n = 0; n < NT; n++)
    {
    output << n * dt << "  " << V2[Eout[0]][Eout[1]] + V4[Eout[0]][Eout[1]] << endl;
    if (n % 100 == 0)
        cout << n << endl;
    }

    // Free memory allocated for devices
    cudaFree(dev_V1);
    cudaFree(dev_V2);
    cudaFree(dev_V3);
    cudaFree(dev_V4);

    output.close();

    // Print execution time
    cout << "Done ";
    std::cout << ((std::clock() - start) / (double)CLOCKS_PER_SEC) << '\n';
    cin.get();

}
