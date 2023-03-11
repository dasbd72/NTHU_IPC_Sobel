// #define DEBUG
#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8
#define TILE_X 8
#define TILE_Y 8
#define numberOfSMs 20
#define numberOfWarps 32
/*
Device 0: "GeForce GTX 1080"
  CUDA Driver Version / Runtime Version          11.0 / 11.0
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 8112 MBytes (8505524224 bytes)
  (20) Multiprocessors, (128) CUDA Cores/MP:     2560 CUDA Cores
  GPU Max Clock rate:                            1835 MHz (1.84 GHz)
  Memory Clock rate:                             5005 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 2097152 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
 */

#ifdef DEBUG
#include <chrono>
#define debug_printf(fmt, args...) printf(fmt, ##args);
#else
#define debug_printf(fmt, args...)
#endif

// clang-format off
__constant__ short mask[MASK_N][MASK_X][MASK_Y] = {
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0},
     {  2,  8, 12,  8,  2},
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1},
     { -4, -8,  0,  8,  4},
     { -6,-12,  0, 12,  6},
     { -4, -8,  0,  8,  4},
     { -1, -2,  0,  2,  1}}
};
// clang-format on

void cudaErrorPrint(cudaError_t err) {
    if (err != cudaSuccess) {
        debug_printf("Error: %s\n", cudaGetErrorString(err));
    }
}

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width,
             unsigned* channels) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char*)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0; i < *height; ++i) {
        row_pointers[i] = *image + i * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void sobel(const unsigned char* const src_img, unsigned char* const dst_img, const unsigned height, const unsigned width, const unsigned channels) {
    __shared__ short shared_mask[MASK_N][MASK_X][MASK_Y];
    __shared__ unsigned char shared_src_img[TILE_X + MASK_X - 1][TILE_Y + MASK_Y - 1][3];
    // copy convolution weight matrix to shared memory
    int thread_index = threadIdx.x + blockDim.x * threadIdx.y;
    int thread_mask_n = thread_index / (MASK_X * MASK_Y);
    int thread_mask_x = thread_index / MASK_Y % MASK_X;
    int thread_mask_y = thread_index % MASK_Y;
    if (thread_mask_n < MASK_N && thread_mask_x < MASK_X && thread_mask_y < MASK_Y)
        shared_mask[thread_mask_n][thread_mask_x][thread_mask_y] = mask[thread_mask_n][thread_mask_x][thread_mask_y];
    // copy source image to shared memory
    int start_x = blockIdx.x * blockDim.x;
    int start_y = blockIdx.y * blockDim.y;
    for (int di = thread_index; di < (TILE_X + MASK_X - 1) * (TILE_Y + MASK_Y - 1); di += TILE_X * TILE_Y) {
        int dx = di / (TILE_X + MASK_X - 1) - (MASK_X / 2);
        int dy = di % (TILE_Y + MASK_Y - 1) - (MASK_Y / 2);
        if (start_x + dx >= 0 && start_x + dx < width && start_y + dy >= 0 && start_y + dy < height) {
            shared_src_img[dx + (MASK_X / 2)][dy + (MASK_Y / 2)][2] = src_img[channels * (width * (start_y + dy) + start_x + dx) + 2];
            shared_src_img[dx + (MASK_X / 2)][dy + (MASK_Y / 2)][1] = src_img[channels * (width * (start_y + dy) + start_x + dx) + 1];
            shared_src_img[dx + (MASK_X / 2)][dy + (MASK_Y / 2)][0] = src_img[channels * (width * (start_y + dy) + start_x + dx) + 0];
        }
    }
    // sync the threads
    __syncthreads();

    int x = start_x + threadIdx.x;
    int y = start_y + threadIdx.y;
    if (x < width && y < height) {
        int val[MASK_N * 3] = {0};
        for (short i = 0; i < MASK_N; ++i) {
            const short adjustX = (MASK_X % 2) ? 1 : 0;
            const short adjustY = (MASK_Y % 2) ? 1 : 0;
            const short xBound = (MASK_X / 2);
            const short yBound = (MASK_Y / 2);

            val[i * 3 + 2] = 0;
            val[i * 3 + 1] = 0;
            val[i * 3] = 0;

            for (short v = -yBound; v < yBound + adjustY; ++v) {
                for (short u = -xBound; u < xBound + adjustX; ++u) {
                    if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
                        unsigned char R = shared_src_img[(int)threadIdx.x + (MASK_X / 2) + u][(int)threadIdx.y + (MASK_Y / 2) + v][2];
                        unsigned char G = shared_src_img[(int)threadIdx.x + (MASK_X / 2) + u][(int)threadIdx.y + (MASK_Y / 2) + v][1];
                        unsigned char B = shared_src_img[(int)threadIdx.x + (MASK_X / 2) + u][(int)threadIdx.y + (MASK_Y / 2) + v][0];
                        val[i * 3 + 2] += R * shared_mask[i][u + xBound][v + yBound];
                        val[i * 3 + 1] += G * shared_mask[i][u + xBound][v + yBound];
                        val[i * 3 + 0] += B * shared_mask[i][u + xBound][v + yBound];
                    }
                    __syncthreads();
                }
            }
        }

        float totalR = 0.0;
        float totalG = 0.0;
        float totalB = 0.0;
        for (short i = 0; i < MASK_N; ++i) {
            totalR += val[i * 3 + 2] * val[i * 3 + 2];
            totalG += val[i * 3 + 1] * val[i * 3 + 1];
            totalB += val[i * 3 + 0] * val[i * 3 + 0];
        }

        totalR = sqrtf(totalR) / SCALE;
        totalG = sqrtf(totalG) / SCALE;
        totalB = sqrtf(totalB) / SCALE;
        const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
        dst_img[channels * (width * y + x) + 2] = cR;
        dst_img[channels * (width * y + x) + 1] = cG;
        dst_img[channels * (width * y + x) + 0] = cB;
    }
}

int main(int argc, char** argv) {
    assert(argc == 3);

    int deviceId;

    cudaGetDevice(&deviceId);
    debug_printf("[DBG] deviceId: %d\n", deviceId);

    unsigned height, width, channels;
    unsigned char* src_img = NULL;
    unsigned char* dst_img;
    unsigned char* device_src_img;
    unsigned char* device_dst_img;
    size_t img_size;

#ifdef DEBUG
    auto start1 = std::chrono::high_resolution_clock::now();
#endif
    read_png(argv[1], &src_img, &height, &width, &channels);  // Load source image
#ifdef DEBUG
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    debug_printf("[DBG] read_png duration: %d ms\n", duration1.count() / 1000);
#endif
    assert(channels == 3);                                         // Chech if there are three channels
    img_size = height * width * channels * sizeof(unsigned char);  // Set destinated image
    debug_printf("[DBG] height: %d\n", height);
    debug_printf("[DBG] width: %d\n", width);
    debug_printf("[DBG] channels: %d\n", channels);

    cudaErrorPrint(cudaMalloc(&device_src_img, img_size));  // Cuda malloc gpu source image
    // cudaErrorPrint(cudaMemcpy(device_src_img, src_img, img_size, cudaMemcpyHostToDevice));  // Cuda copy source image
    cudaErrorPrint(cudaMemcpyAsync(device_src_img, src_img, img_size, cudaMemcpyHostToDevice));  // Cuda asyncronize copy source image
    cudaErrorPrint(cudaMalloc(&device_dst_img, img_size));                                       // Malloc Cuda destinated image

    dim3 myBlockDim(TILE_X, TILE_Y);
    dim3 myGridDim(ceil((float)width / TILE_X), ceil((float)height / TILE_Y));
    debug_printf("[DBG] myGridDim: (%d, %d, %d)\n", myGridDim.x, myGridDim.y, myGridDim.z);
    debug_printf("[DBG] myBlockDim: (%d, %d, %d)\n", myBlockDim.x, myBlockDim.y, myBlockDim.z);
    sobel<<<myGridDim, myBlockDim>>>(device_src_img, device_dst_img, height, width, channels);  // Start sobel

    dst_img = (unsigned char*)malloc(img_size);  // malloc destinated image
    // cudaErrorPrint(cudaMemcpy(dst_img, device_dst_img, img_size, cudaMemcpyDeviceToHost));  // Cuda copy destinated image
    cudaErrorPrint(cudaMemcpyAsync(dst_img, device_dst_img, img_size, cudaMemcpyDeviceToHost));  // Cuda asyncronize copy destinated image
    cudaErrorPrint(cudaGetLastError());                                                          // Cuda debug
    cudaErrorPrint(cudaDeviceSynchronize());                                                     // Cuda synchronize
#ifdef DEBUG
    auto start2 = std::chrono::high_resolution_clock::now();
#endif
    write_png(argv[2], dst_img, height, width, channels);  // Write File
#ifdef DEBUG
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    debug_printf("[DBG] write_png duration: %d ms\n", duration2.count() / 1000);
#endif

    free(src_img);
    free(dst_img);
    cudaFree(device_src_img);
    cudaFree(device_dst_img);

    debug_printf("[DBG] Program Ended.\n");
    return 0;
}

/*
make; srun -p ipc22 --gres=gpu:1 nvprof ./hw3 samples/1.png outputs/1.out.png; ./hw3-diff samples/1.out.png outputs/1.out.png
make; srun -p ipc22 --gres=gpu:1 nvprof ./hw3 samples/2.png outputs/2.out.png; ./hw3-diff samples/2.out.png outputs/2.out.png
make; srun -p ipc22 --gres=gpu:1 nvprof ./hw3 samples/3.png outputs/3.out.png; ./hw3-diff samples/3.out.png outputs/3.out.png
make; srun -p ipc22 --gres=gpu:1 nvprof ./hw3 samples/4.png outputs/4.out.png; ./hw3-diff samples/4.out.png outputs/4.out.png
make; srun -p ipc22 --gres=gpu:1 nvprof ./hw3 samples/5.png outputs/5.out.png; ./hw3-diff samples/5.out.png outputs/5.out.png
make; srun -p ipc22 --gres=gpu:1 nvprof ./hw3 samples/6.png outputs/6.out.png; ./hw3-diff samples/6.out.png outputs/6.out.png
make; srun -p ipc22 --gres=gpu:1 nvprof ./hw3 samples/7.png outputs/7.out.png; ./hw3-diff samples/7.out.png outputs/7.out.png
make; srun -p ipc22 --gres=gpu:1 nvprof ./hw3 samples/8.png outputs/8.out.png; ./hw3-diff samples/8.out.png outputs/8.out.png

make; srun -p ipc22 --gres=gpu:1 ./hw3 samples/1.png outputs/1.out.png; ./hw3-diff samples/1.out.png outputs/1.out.png
make; srun -p ipc22 --gres=gpu:1 ./hw3 samples/2.png outputs/2.out.png; ./hw3-diff samples/2.out.png outputs/2.out.png
make; srun -p ipc22 --gres=gpu:1 ./hw3 samples/3.png outputs/3.out.png; ./hw3-diff samples/3.out.png outputs/3.out.png
make; srun -p ipc22 --gres=gpu:1 ./hw3 samples/4.png outputs/4.out.png; ./hw3-diff samples/4.out.png outputs/4.out.png
make; srun -p ipc22 --gres=gpu:1 ./hw3 samples/5.png outputs/5.out.png; ./hw3-diff samples/5.out.png outputs/5.out.png
make; srun -p ipc22 --gres=gpu:1 ./hw3 samples/6.png outputs/6.out.png; ./hw3-diff samples/6.out.png outputs/6.out.png
make; srun -p ipc22 --gres=gpu:1 ./hw3 samples/7.png outputs/7.out.png; ./hw3-diff samples/7.out.png outputs/7.out.png
make; srun -p ipc22 --gres=gpu:1 ./hw3 samples/8.png outputs/8.out.png; ./hw3-diff samples/8.out.png outputs/8.out.png

make; srun -p ipc22 --gres=gpu:1 ./hw3 view.png view.out.png
make; srun -p ipc22 --gres=gpu:1 nvprof ./hw3 view.png view.out.png
srun -p ipc22 --gres=gpu:1 nvprof -f -o  analysis.nvprof ./hw3 samples/2.png outputs/2.out.png

 */