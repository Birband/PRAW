#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
// Constants
const double G = 6.67e-11;
const double TIMESTEP = 1.0;
const int NUM_STEPS = 10;
const int NUM_PARTICLES = 10000000;
const double EPSILON = 0.5;
const double THETA = 0.7;

// Particle structure
struct Particle {
    double x, y, fx, fy, vx, vy, mass;
};

// Node structure for quadtree
struct Node {
    double x_min, x_max, y_min, y_max;
    double com_x, com_y, total_mass;
    int children[4];
    int particle_idx;
    bool is_leaf;

    __device__ Node() : x_min(0), x_max(0), y_min(0), y_max(0),
                        com_x(0), com_y(0), total_mass(0),
                        particle_idx(-1), is_leaf(true) {
        children[0] = children[1] = children[2] = children[3] = -1;
    }
};

// Kernel prototypes
__global__ void resetTreeKernel(Node* tree, int* node_count);
__global__ void insertParticlesKernel(Particle* particles, Node* tree, int* node_count, int max_nodes);
__global__ void computeForcesKernel(Particle* particles, Node* tree, double theta);
__global__ void updatePositionsKernel(Particle* particles, double dt);
__global__ void calculateBounds(Node* tree, int* node_count);

// Helper functions
void generateParticles(Particle* particles) {
    // Central massive particle
    particles[0] = {0, 0, 0, 0, 0, 0, 1e18};

    // Generate orbiting particles
    for (int i = 1; i < NUM_PARTICLES; ++i) {
        double theta = 2 * M_PI * rand() / RAND_MAX;
        double r = 3e3 + (4e4 - 3e3) * (rand() / (double)RAND_MAX);
        double mass = 1e10 + (1e12 - 1e10) * (rand() / (double)RAND_MAX);
        double orbital_v = sqrt(G * 1e18 / r);
        
        particles[i] = {
            r * cos(theta),           // x
            r * sin(theta),           // y
            0, 0,                     // fx, fy
            -orbital_v * sin(theta),  // vx
            orbital_v * cos(theta),   // vy
            mass                      // mass
        };
    }
}

int main() {
    // Allocate Unified Memory
    Particle* d_particles;
    Node* d_tree;
    int* d_node_count;
    
    cudaMallocManaged(&d_particles, NUM_PARTICLES * sizeof(Particle));
    cudaMallocManaged(&d_tree, 10 * NUM_PARTICLES * sizeof(Node)); // Oversized tree
    cudaMallocManaged(&d_node_count, sizeof(int));

    // Generate particles on CPU
    generateParticles(d_particles);
    std::chrono::high_resolution_clock::time_point start, end;
    // Simulation loop
    for (int step = 0; step < NUM_STEPS; ++step) {
        start = std::chrono::high_resolution_clock::now();
        calculateBounds<<<1, 1>>>(d_tree, d_node_count);
        cudaDeviceSynchronize();
        // Reset tree
        resetTreeKernel<<<1, 1>>>(d_tree, d_node_count);
        cudaDeviceSynchronize();

        // Build tree
        insertParticlesKernel<<<(NUM_PARTICLES + 255)/256, 256>>>(
            d_particles, d_tree, d_node_count, 10 * NUM_PARTICLES
        );
        cudaDeviceSynchronize();

        // Compute forces
        computeForcesKernel<<<(NUM_PARTICLES + 255)/256, 256>>>(
            d_particles, d_tree, THETA
        );
        cudaDeviceSynchronize();

        // Update positions
        updatePositionsKernel<<<(NUM_PARTICLES + 255)/256, 256>>>(
            d_particles, TIMESTEP
        );
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() * 1000;
        std::printf("Lap: %i Total time: %f ms\n", step + 1, total_time);
    }

    // Cleanup
    cudaFree(d_particles);
    cudaFree(d_tree);
    cudaFree(d_node_count);

    return 0;
}

// Tree management kernels
__global__ void resetTreeKernel(Node* tree, int* node_count) {
    tree[0] = Node();  // Reset root node
    tree[0].x_min = -4e4; tree[0].x_max = 4e4;
    tree[0].y_min = -4e4; tree[0].y_max = 4e4;
    *node_count = 1;
}

__global__ void insertParticlesKernel(Particle* particles, Node* tree, int* node_count, int max_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    Particle p = particles[idx];
    int current_node = 0;

    while (true) {
        Node* node = &tree[current_node];
        
        if (node->is_leaf) {
            if (node->particle_idx == -1) {
                // Empty leaf node
                node->particle_idx = idx;
                node->total_mass = p.mass;
                node->com_x = p.x;
                node->com_y = p.y;
                break;
            } else {
                // Split node
                double x_mid = (node->x_min + node->x_max) / 2;
                double y_mid = (node->y_min + node->y_max) / 2;
                
                // Create children
                for (int i = 0; i < 4; i++) {
                    int child_idx = atomicAdd(node_count, 1);
                    if (child_idx >= max_nodes) return;
                    
                    tree[child_idx] = Node();
                    tree[child_idx].x_min = (i & 1) ? x_mid : node->x_min;
                    tree[child_idx].x_max = (i & 1) ? node->x_max : x_mid;
                    tree[child_idx].y_min = (i & 2) ? y_mid : node->y_min;
                    tree[child_idx].y_max = (i & 2) ? node->y_max : y_mid;
                    node->children[i] = child_idx;
                }
                
                node->is_leaf = false;
                int old_particle = node->particle_idx;
                node->particle_idx = -1;
                
                // Reinsert old particle
                Particle old_p = particles[old_particle];
                for (int i = 0; i < 4; i++) {
                    Node* child = &tree[node->children[i]];
                    if (old_p.x >= child->x_min && old_p.x <= child->x_max &&
                        old_p.y >= child->y_min && old_p.y <= child->y_max) {
                        child->particle_idx = old_particle;
                        child->total_mass = old_p.mass;
                        child->com_x = old_p.x;
                        child->com_y = old_p.y;
                        break;
                    }
                }
            }
        } else {
            // Update COM and mass
            double total_mass = node->total_mass + p.mass;
            node->com_x = (node->com_x * node->total_mass + p.x * p.mass) / total_mass;
            node->com_y = (node->com_y * node->total_mass + p.y * p.mass) / total_mass;
            node->total_mass = total_mass;

            // Find appropriate child
            double x_mid = (node->x_min + node->x_max) / 2;
            double y_mid = (node->y_min + node->y_max) / 2;
            int child_idx = 0;
            if (p.x > x_mid) child_idx |= 1;
            if (p.y > y_mid) child_idx |= 2;
            current_node = node->children[child_idx];
        }
    }
}

// Physics kernels
__global__ void computeForcesKernel(Particle* particles, Node* tree, double theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    Particle* p = &particles[idx];
    p->fx = p->fy = 0.0;

    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        Node* node = &tree[node_idx];

        if (node_idx != 0 && node->is_leaf) {
            if (node->particle_idx != idx) {
                Particle* other = &particles[node->particle_idx];
                double dx = other->x - p->x;
                double dy = other->y - p->y;
                double r = sqrt(dx*dx + dy*dy + EPSILON*EPSILON);
                double f = G * p->mass * other->mass / (r*r);
                p->fx += f * dx / r;
                p->fy += f * dy / r;
            }
        } else {
            double dx = node->com_x - p->x;
            double dy = node->com_y - p->y;
            double r = sqrt(dx*dx + dy*dy);
            double s = node->x_max - node->x_min;

            if (s / r < theta || node->is_leaf) {
                if (r > 0) {
                    double f = G * p->mass * node->total_mass / (r*r + EPSILON*EPSILON);
                    p->fx += f * dx / r;
                    p->fy += f * dy / r;
                }
            } else {
                for (int i = 0; i < 4; i++) {
                    if (node->children[i] != -1) {
                        stack[stack_ptr++] = node->children[i];
                    }
                }
            }
        }
    }
}

__global__ void updatePositionsKernel(Particle* particles, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    Particle* p = &particles[idx];
    p->vx += p->fx / p->mass * dt;
    p->vy += p->fy / p->mass * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    p->fx = p->fy = 0.0;
}

__global__ void calculateBounds(Node* tree, int* node_count) {
    for (int i = 0; i < *node_count; i++) {
        Node* node = &tree[i];
        if (node->is_leaf) continue;

        node->com_x = 0;
        node->com_y = 0;
        node->total_mass = 0;

        for (int j = 0; j < 4; j++) {
            int child_idx = node->children[j];
            if (child_idx != -1) {
                Node* child = &tree[child_idx];
                node->com_x += child->com_x * child->total_mass;
                node->com_y += child->com_y * child->total_mass;
                node->total_mass += child->total_mass;
            }
        }

        if (node->total_mass > 0) {
            node->com_x /= node->total_mass;
            node->com_y /= node->total_mass;
        }
    }
}