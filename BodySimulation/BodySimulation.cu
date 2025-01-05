#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstdio>

const double G = 6.67e-11;
const double TIMESTEP = 1;
const int NUM_STEPS = 10;
const int NUMBER_OF_PARTICLES = 100000;

const double MAX_RADIUS = 5e4;
const double MIN_RADIUS = 3e3;
const double MAX_MASS = 1e12;
const double MIN_MASS = 1e10;
const double CENTRAL_MASS = 1e20;

struct Particle {
    double x, y;
    double fx, fy;
    double vx, vy;
    double mass;

    __host__ __device__ Particle(double mass, double x, double y, double vx, double vy)
        : mass(mass), x(x), y(y), vx(vx), vy(vy), fx(0), fy(0) {}
};

__global__ void calculateForceCUDA(Particle* particles, int num_particles, double G) {
    extern __shared__ Particle shared_particles[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& p1 = particles[i];
    p1.fx = 0;
    p1.fy = 0;

    for (int tile = 0; tile < (num_particles + blockDim.x - 1) / blockDim.x; ++tile) {
        int index = tile * blockDim.x + threadIdx.x;

        if (index < num_particles) {
            shared_particles[threadIdx.x] = particles[index];
        }
        __syncthreads();

        for (int j = 0; j < blockDim.x; ++j) {
            if (tile * blockDim.x + j >= num_particles || i == tile * blockDim.x + j) continue;

            Particle& p2 = shared_particles[j];
            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            double epsilon = 0.5;
            double r = sqrt(dx * dx + dy * dy + epsilon * epsilon);

            double force = (G * p1.mass * p2.mass) / (r * r + epsilon * epsilon);
            p1.fx += force * dx / r;
            p1.fy += force * dy / r;
        }
        __syncthreads();
    }
}

__global__ void updatePositionCUDA(Particle* particles, int num_particles, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& p = particles[i];
    p.vx += p.fx / p.mass * dt;
    p.vy += p.fy / p.mass * dt;

    p.x += p.vx * dt;
    p.y += p.vy * dt;

    p.fx = 0;
    p.fy = 0;
}

void generateParticles(std::vector<Particle>& particles, int num_particles) {
    particles.push_back(Particle(CENTRAL_MASS, 0, 0, 0, 0));
    for (int i = 0; i < num_particles; ++i) {
        double theta = 2 * M_PI * rand() / RAND_MAX;
        double r = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * (rand() / double(RAND_MAX));
        double x = r * cos(theta);
        double y = r * sin(theta);

        double mass = MIN_MASS + (MAX_MASS - MIN_MASS) * (rand() / double(RAND_MAX));

        double velocity_scaling = 1.0;
        double orbital_velocity = sqrt(G * CENTRAL_MASS / r) * velocity_scaling;

        double vx = -y * orbital_velocity / r;
        double vy = x * orbital_velocity / r;

        particles.push_back(Particle(mass, x, y, vx, vy));
    }
}

void runSimulation(std::vector<Particle>& particles, int num_particles, double G, double dt) {
    Particle* d_particles;
    cudaMalloc(&d_particles, num_particles * sizeof(Particle));

    cudaMemcpy(d_particles, particles.data(), num_particles * sizeof(Particle), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_particles + blockSize - 1) / blockSize;

    size_t sharedMemSize = blockSize * sizeof(Particle);

    for (int step = 0; step < NUM_STEPS; ++step) {
        auto start = std::chrono::high_resolution_clock::now();

        calculateForceCUDA<<<gridSize, blockSize, sharedMemSize>>>(d_particles, num_particles, G);
        cudaDeviceSynchronize();
        updatePositionCUDA<<<gridSize, blockSize>>>(d_particles, num_particles, dt);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() * 1000;
        std::printf("Lap: %i Total time: %f ms\n", step + 1, total_time);
    }

    cudaMemcpy(particles.data(), d_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
}

int main() {
    std::vector<Particle> particles;
    generateParticles(particles, NUMBER_OF_PARTICLES);

    runSimulation(particles, NUMBER_OF_PARTICLES, G, TIMESTEP);

    return 0;
}
