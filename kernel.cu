// parallel & host code referenced from here: 
// https://arxiv.org/ftp/arxiv/papers/1305/1305.4365.pdf
// https://www.moreware.org/wp/blog/2019/08/27/cuda-with-jetson-nano-parallel-pollard-rho-test/

// serial code referenced from here: 
//https://www.geeksforgeeks.org/pollards-rho-algorithm-prime-factorization/

#include <time.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <chrono>
#include <string>

using namespace std;

__host__ long long int gcd(long long int a, long long int b)
{
    if (a == 0)
        return b;
    return gcd(b % a, a);
}

/* Function to calculate (base^exponent)%modulus */
__host__ long long int modular_pow(long long int base, int exponent,
    long long int modulus)
{
    /* initialize result */
    long long int result = 1;

    while (exponent > 0)
    {
        /* if y is odd, multiply base with result */
        if (exponent & 1)
            result = (result * base) % modulus;

        /* exponent = exponent/2 */
        exponent = exponent >> 1;

        /* base = base * base */
        base = (base * base) % modulus;
    }
    return result;
}

/* method to return prime divisor for n */
__host__ long long int PollardRho(long long int n)
{
    /* initialize random seed */
    srand(time(NULL));

    /* no prime divisor for 1 */
    if (n == 1) return n;

    /* even number means one of the divisors is 2 */
    if (n % 2 == 0) return 2;

    /* we will pick from the range [2, N) */
    long long int x = (rand() % (n - 2)) + 2;
    long long int y = x;

    /* the constant in f(x).
    * Algorithm can be re-run with a different c
    * if it throws failure for a composite. */
    long long int c = (rand() % (n - 1)) + 1;

    /* Initialize candidate divisor (or result) */
    long long int d = 1;

    /* until the prime factor isn't obtained.
    If n is prime, return n */
    while (d == 1)
    {
        /* Tortoise Move: x(i+1) = f(x(i)) */
        x = (modular_pow(x, 2, n) + c + n) % n;

        /* Hare Move: y(i+1) = f(f(y(i))) */
        y = (modular_pow(y, 2, n) + c + n) % n;
        y = (modular_pow(y, 2, n) + c + n) % n;

        /* check gcd of |x-y| and n */
        d = gcd((abs(x - y)), n);

        /* retry if the algorithm fails to find prime factor
        * with chosen x and c */
        if (d == n) return PollardRho(n);
    }

    return d;
}

// host version of f(x)
__host__ __forceinline__ long long int fx(long long int x, long long int a, long long int c) {
    return (a * x * x + c);
}

// host version of a binary gcd algorithm
__host__ long long int gcd_h(long long int u, long long int v)
{
    int shift;

    /* GCD(0,v) == v; GCD(u,0) == u, GCD(0,0) == 0 */
    if (u == 0) return v;
    if (v == 0) return u;

    /* Let shift := lg K, where K is the greatest power of 2
     dividing both u and v. */
    for (shift = 0; ((u | v) & 1) == 0; ++shift) {
        u >>= 1;
        v >>= 1;
    }

    while ((u & 1) == 0)
        u >>= 1;

    /* From here on, u is always odd. */
    do {
        /* remove all factors of 2 in v -- they are not common */
        /*   note: v is not zero, so while will terminate */
        while ((v & 1) == 0)  /* Loop X */
            v >>= 1;

        /* Now u and v are both odd. Swap if necessary so u <= v,
         then set v = v - u (which is even). For bignums, the
         swapping is just pointer movement, and the subtraction
         can be done in-place. */
        if (u > v) {
            long long int t = v; v = u; u = t;
        }  // Swap u and v.
        v = v - u;                       // Here v >= u.
    } while (v != 0);

    /* restore common factors of 2 */
    return u << shift;
}

// host version of the Pollard's Rho algorithm
__host__ long long int pollardHost(long long int num)
{


    long long int max = sqrt(num);

    // catch easy cases
    if (num % 2 == 0)
    {
        //       cout << "Found 2" << endl;
        return 2;

    }
    else if (num % 3 == 0)
    {
        return 3;
    }
    else if (max * max == num)
    {
        return max;
    }

    long long int result = 0;
    bool quit = false;

    long long int x = 0;
    long long int a = rand() % (max - 1) + 1;
    long long int c = rand() % (max - 1) + 1;
    long long int y, d, z;

    y = x;
    d = 1;

    do
    {
        x = fx(x, a, c) % num;
        y = fx(fx(y, a, c), a, c) % num;
        z = std::abs(x - y);
        d = gcd(z, num);
    } while (d == 1 && !quit);


    if (d != 1 && d != num)
    {
        quit = true;
        result = d;
    }

    return result;
}

// device version of f(x)
__device__ __forceinline__ long long int fx_d(long long int x, long long int a, long long int c) {
    return (a * x * x + c);
}

// device version of binary gcd
__device__ long long int gcd_d(long long int u, long long int v)
{
    int shift;

    /* GCD(0,v) == v; GCD(u,0) == u, GCD(0,0) == 0 */
    if (u == 0) return v;
    if (v == 0) return u;

    /* Let shift := lg K, where K is the greatest power of 2
     dividing both u and v. */
    for (shift = 0; ((u | v) & 1) == 0; ++shift) {
        u >>= 1;
        v >>= 1;
    }

    while ((u & 1) == 0)
        u >>= 1;

    /* From here on, u is always odd. */
    do {
        /* remove all factors of 2 in v -- they are not common */
        /*   note: v is not zero, so while will terminate */
        while ((v & 1) == 0)  /* Loop X */
            v >>= 1;

        /* Now u and v are both odd. Swap if necessary so u <= v,
         then set v = v - u (which is even). For bignums, the
         swapping is just pointer movement, and the subtraction
         can be done in-place. */
        if (u > v) {
            long long int t = v; v = u; u = t;
        }  // Swap u and v.
        v = v - u;                       // Here v >= u.
    } while (v != 0);

    /* restore common factors of 2 */
    return u << shift;
}

// CUDA kernel for Pollard's Rho
// Only execute a single pass
__global__ void pollardKernel(long long int num, long long int* xd, long long int* result)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    long long int x, y, a, c, d, z;
    d = 1;

    // copy state variables back into local memory
    x = xd[threadID * 4];
    y = xd[threadID * 4 + 1];
    a = xd[threadID * 4 + 2];
    c = xd[threadID * 4 + 3];

    // execute the pass
    x = fx_d(x, a, c) % num;
    y = fx_d(fx_d(y, a, c), a, c) % num;
    z = abs(x - y);
    d = gcd_d(z, num);

    // copy updated state back into global memory
    xd[threadID * 4] = x;
    xd[threadID * 4 + 1] = y;

    // test to see if it found a factor
    if (d != 1 && d != num)
    {
        // if found, copy it into global syncronization variable "found"
        *result = d;
    }

}

// wrapper that sets up and calls pollardKernel
long long int pollardDevice(long long int num, int blocksPerGrid, int threadsPerBlock)
{
    //  Calculate matrix dimensions
    int n = 4 * blocksPerGrid * threadsPerBlock;
    int N = n * sizeof(long long int);

    // local variables
    long long int max = (long long int) sqrt(num);
    long long int result = 0;

    // catch easy cases
    if (num % 2 == 0)
    {
        return 2;

    }
    else if (num % 3 == 0)
    {
        return 3;
    }
    else if (max * max == num)
    {
        return max;
    }

    // initialize the state array
    long long int* x;
    x = (long long int*)malloc(N);
    if (!x) cout << "Could not allocate host memory\n";


    for (int i = 0; i < n; i += 4)
    {
        //x[i] = rand() % max + 1;
        //x[1 + 1] = x[i];

        // set x, y, a, and c for each thread
        x[i] = 0;
        x[i + 1] = 0;
        x[i + 2] = rand() % (max - 1) + 1;
        x[i + 3] = rand() % (max - 1) + 1;

    }
    // Allocate device memory
    long long int* result_d;
    if (cudaMalloc((void**)&result_d, sizeof(long long int))) cout << "Cannot allocate device memory result_d\n";
    if (cudaMemcpy(result_d, &result, sizeof(long long int), cudaMemcpyHostToDevice)) cout << "Cannot copy result from host to device\n";

    long long int* Xd;
    if (cudaMalloc((void**)&Xd, N)) cout << "Cannot allocate device memory Ad\n";
    // do an asychronous copy operation and let the CUDA runtime sort out the details
    if (cudaMemcpyAsync(Xd, x, N, cudaMemcpyHostToDevice)) cout << "Cannot copy X from host to device\n";



    // run the kernel until it finds a result
    do {
        pollardKernel << <blocksPerGrid, threadsPerBlock >> > (num, Xd, result_d);

        cudaMemcpy(&result, result_d, sizeof(long long int), cudaMemcpyDeviceToHost);

    } while (result == 0);



    // if it failed, abort
    if (cudaGetLastError()) cout << "pollardKernel failed\n";

    //  Free device memory
    cudaFree(Xd);
    cudaFree(result_d);
    return result;
}

int main()
{
    int num;
    cudaGetDeviceCount(&num);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int Gflops = prop.multiProcessorCount * prop.clockRate;
    printf("CUDA Device %d: %s Gflops %f Processors %d Threads/Block %d\n\n", num, prop.name, 1e-6 * Gflops, prop.multiProcessorCount, prop.maxThreadsPerBlock);
    std::string instream="";
    ifstream infile;
    int length = 0;
    // { 27457082047, 70129855908257459, 24803950017422581, 277211660740868813, 400022021097189157, 597384463925502007 };
    long long int N[50];
    // file location to primes list
    infile.open("C:/Users/Trang/Documents/Class/Spring_2020/AdvancedCrypto/CryptoFinal/primes32.txt");
    if (infile.is_open())
    {
        srand(time(NULL));
        int findex = 0;
        infile.seekg(0, infile.end);
        length = infile.tellg();
        infile.seekg(0, infile.beg);
        cout << "N[] = {\n";
        for (int i = 0; i < 50; i++)
        {
            findex = rand();
            while (findex < length - 20 && (infile.seekg(findex - 1).peek() != ' '))
                findex--;
            if (infile.seekg(findex).peek() == ' ')
                findex++;
            infile.seekg(findex);
            infile >> instream;
            //cout << instream << endl;
            //cout << "Instream val: " << instream << endl;
            long long int temp = stoll(instream, nullptr);
            N[i] = temp;
            cout << "p: " << temp << "\t";
            instream.clear();
            findex += 10;
            infile >> instream;
            temp = stoll(instream, nullptr);
            cout << "q: " << temp << "\t";
            instream.clear();
            N[i] *= temp;
            cout << "N: " << N[i] << endl;
        }
        cout << "}\n\n";

        int blocksPerGrid = 512;
        int threadsPerBlock = 512;
        cout << "Numbr of blocks: " << blocksPerGrid << "\tNumber of threads/block: " << threadsPerBlock << endl;
        int width = blocksPerGrid * threadsPerBlock;

        using clock = std::chrono::system_clock;
        using sec = std::chrono::duration<double>;
        // for milliseconds, use using ms = std::chrono::duration<double, std::milli>;

        const auto before3 = clock::now();
        for (int i = 0; i < sizeof(N) / sizeof(long long int); i++)
        {
            printf("Parallel: One of the divisors for %lld is %lld.\n", N[i], pollardDevice(N[i], blocksPerGrid, threadsPerBlock));
        }
        const sec duration3 = clock::now() - before3;
        cout << "Parallel Pollard Rho took " << duration3.count() << "s\n\n" << endl;

        const auto before1 = clock::now();
        for (int i = 0; i < sizeof(N) / sizeof(long long int); i++)
        {
            printf("Geeks Host/Serial: One of the divisors for %lld is %lld.\n",
                N[i], PollardRho(N[i]));
        }
        const sec duration1 = clock::now() - before1;
        cout << "Geeks Serial Pollard Rho took " << duration1.count() << "s\n\n" << endl;


        const auto before2 = clock::now();
        for (int i = 0; i < sizeof(N) / sizeof(long long int); i++)
        {
            printf("Host/Serial: One of the divisors for %lld is %lld.\n", N[i], pollardHost(N[i]));
        }
        const sec duration2 = clock::now() - before2;
        cout << "Host/Serial Pollard Rho took " << duration2.count() << "s\n\n" << endl;

    }
    else
        cout << "Sorry, couldn't open primes list!" << endl;
    infile.close();

    return 0;
}
