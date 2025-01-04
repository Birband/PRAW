#include <omp.h>
#include <vector>
#include <cmath>
#include <array>
#include <cstdlib> // rand
#include <chrono>
#include <cstdio>

// Simulation parameters
const double G = 6.67e-11;
const double TIMESTEP = 1;
const int NUM_STEPS = 10;
const int NUMBER_OF_PARTICLES = 100000;
const double EPSILON = 0.5;

// Particle generation parameters
const double MAX_RADIUS = 4e4;
const double MIN_RADIUS = 3e3;
const double MAX_MASS = 1e12;
const double MIN_MASS = 1e10;
const double CENTRAL_MASS = 1e18;

// Rendering parameters
const double SCALE = 1e-2;

#define WINDOW_SIZE 1200

struct Particle {
    double x, y;
    double fx, fy;
    double vx, vy;
    double mass;

    Particle(double mass, double x, double y, double vx, double vy)
        : mass(mass), x(x), y(y), vx(vx), vy(vy), fx(0), fy(0) {}
};

double calculateParticleSize(double mass) {
    if (mass > 1e15) return 10;
    else return 1.0;
}

void generateParticles(std::vector<Particle>& particles, int num_particles) {
    particles.push_back(Particle(CENTRAL_MASS, 0, 0, 0, 0));
    for (int i = 0; i < num_particles; ++i) {
        double theta = 2 * M_PI * rand() / RAND_MAX;
        double r = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * (rand() / double(RAND_MAX));
        double x = r * std::cos(theta);
        double y = r * std::sin(theta);

        double mass = MIN_MASS + (MAX_MASS - MIN_MASS) * (rand() / double(RAND_MAX));

        double velocity_scaling = 1.0; 
        double orbital_velocity = std::sqrt(G * CENTRAL_MASS / r) * velocity_scaling;

        double vx = -y * orbital_velocity / r;
        double vy = x * orbital_velocity / r;

        particles.push_back(Particle(mass, x, y, vx, vy));
    }
}

void updateParticles(std::vector<Particle>& particles) {
    #pragma omp parallel for schedule(dynamic)
    for (Particle& particle : particles) {
        double acceleration_x = particle.fx / particle.mass;
        double acceleration_y = particle.fy / particle.mass;

        particle.vx += acceleration_x * TIMESTEP;
        particle.vy += acceleration_y * TIMESTEP;

        particle.x += particle.vx * TIMESTEP;
        particle.y += particle.vy * TIMESTEP;

        particle.fx = 0;
        particle.fy = 0;
    }
}

void computeForces(std::vector<Particle>& particles) {
    size_t n = particles.size();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; i++) {
        double local_force_x = 0.0, local_force_y = 0.0;

        #pragma omp simd reduction(+:local_force_x, local_force_y)
        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;

            double rx = particles[j].x - particles[i].x;
            double ry = particles[j].y - particles[i].y;

            double r_squared = rx * rx + ry * ry + EPSILON * EPSILON;
            double r_mag = std::sqrt(r_squared);
            double inv_r = 1.0 / r_mag;
            double inv_r_squared = inv_r * inv_r;

            double force_mag = G * particles[i].mass * particles[j].mass * inv_r_squared;

            local_force_x += force_mag * rx * inv_r;
            local_force_y += force_mag * ry * inv_r;
        }

        particles[i].fx = local_force_x;
        particles[i].fy = local_force_y;
    }
}

int main() {
    omp_set_num_threads(20);

    std::vector<Particle> particles;
    generateParticles(particles, NUMBER_OF_PARTICLES);

    int step = 0;
    while (step < NUM_STEPS) {

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        computeForces(particles);
        updateParticles(particles);
        
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_taken = end -  start;
        printf("Take: %i Time taken: %f ms \n", step + 1, time_taken.count() * 1000);   

        step++;
    }
}