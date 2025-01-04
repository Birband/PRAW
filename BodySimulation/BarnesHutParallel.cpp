#include <omp.h>
#include <vector>
#include <cmath>
#include <array>
#include <cstdlib> // rand
#include <chrono>
#include <iostream>
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

class Node {
public:
    double x_min, x_max, y_min, y_max;
    double center_x, center_y;
    double mass;
    std::vector<Particle*> particles;
    Node* children[4] = {nullptr};

    Node(double x_min, double x_max, double y_min, double y_max)
        : x_min(x_min), x_max(x_max), y_min(y_min), y_max(y_max), mass(0), center_x(0), center_y(0) {}

    bool isLeaf() const {
        return children[0] == nullptr;
    }

    void clear() {
        for (int i = 0; i < 4; i++) {
            if (children[i]) {
                children[i]->clear();
                delete children[i];
                children[i] = nullptr;
            }
        }
        particles.clear();
        mass = 0;
        center_x = 0;
        center_y = 0;
    }
};

void divideNode(Node* node);
bool insertParticle(Node* root, Particle* particle);

void calculateForce(Particle* particle, Node* root, double theta) {
    if (root->isLeaf()) {
        if (!root->particles.empty() && root->particles[0] != particle) {
            double dx = root->particles[0]->x - particle->x;
            double dy = root->particles[0]->y - particle->y;
            double r = std::sqrt(dx * dx + dy * dy + EPSILON * EPSILON);
            double force = (G * particle->mass * root->particles[0]->mass) / ((r * r  + (EPSILON * EPSILON)));
            particle->fx += force * dx / r;
            particle->fy += force * dy / r;
        }
    } else {
        double size = std::max(root->x_max - root->x_min, root->y_max - root->y_min);
        double dx = root->center_x - particle->x;
        double dy = root->center_y - particle->y;
        double r = std::sqrt(dx * dx + dy * dy + EPSILON * EPSILON);

        if (size / r < theta) {
            double force = (G * particle->mass * root->particles[0]->mass) / ((r * r  + (EPSILON * EPSILON)));
            particle->fx += force * dx / r;
            particle->fy += force * dy / r;
        } else {
            for (int i = 0; i < 4; i++) {
                if (root->children[i]) {
                    calculateForce(particle, root->children[i], theta);
                }
            }
        }
    }
}

void divideNode(Node* node) {
    double x_mid = (node->x_min + node->x_max) / 2;
    double y_mid = (node->y_min + node->y_max) / 2;

    node->children[0] = new Node(node->x_min, x_mid, node->y_min, y_mid);
    node->children[1] = new Node(x_mid, node->x_max, node->y_min, y_mid);
    node->children[2] = new Node(node->x_min, x_mid, y_mid, node->y_max);
    node->children[3] = new Node(x_mid, node->x_max, y_mid, node->y_max);

    for (Particle* particle : node->particles) {
        for (int i = 0; i < 4; i++) {
            if (insertParticle(node->children[i], particle)) {
                break;
            }
        }
    }

    node->particles.clear();
}

bool insertParticle(Node* root, Particle* particle) {
    if (particle->x < root->x_min || particle->x > root->x_max || particle->y < root->y_min || particle->y > root->y_max) {
        return false;
    }

    if (root->isLeaf()) {
        if (root->particles.empty()) {
            root->particles.push_back(particle);
            root->mass = particle->mass;
            root->center_x = particle->x;
            root->center_y = particle->y;
        } else {
            root->particles.push_back(particle);

            root->mass += particle->mass;
            root->center_x = (root->center_x * (root->mass - particle->mass) + particle->x * particle->mass) / root->mass;
            root->center_y = (root->center_y * (root->mass - particle->mass) + particle->y * particle->mass) / root->mass;

            if (root->particles.size() > 1) {
                divideNode(root);
            }
        }
    } else {
        for (int i = 0; i < 4; i++) {
            if (insertParticle(root->children[i], particle)) {
                break;
            }
        }
    }
    return true;
}

void updatePosition(Particle* p, double dt) {
    p->vx += p->fx / p->mass * dt;
    p->vy += p->fy / p->mass * dt;

    p->x += p->vx * dt;
    p->y += p->vy * dt;

    p->fx = 0;
    p->fy = 0;
}

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

void runSimulation(std::vector<Particle>& particles, Node* root, double theta, double dt) {
    root->clear();
    #pragma omp parallel for schedule(static)
    for (Particle& p : particles) {
        p.fx = 0;
        p.fy = 0;
    }
    
    for (Particle& p : particles) {
        insertParticle(root, &p);
    }

    #pragma omp parallel for schedule(dynamic)
    for (Particle& p : particles) {
        calculateForce(&p, root, theta);
    }

    #pragma omp parallel for schedule(static)
    for (Particle& p : particles) {
        updatePosition(&p, dt);
    }
}

int main() {
    omp_set_num_threads(20);

    std::vector<Particle> particles;
    generateParticles(particles, NUMBER_OF_PARTICLES);

    Node* root = new Node(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS);

    int step = 0;
    while (step < NUM_STEPS) {

        auto start = std::chrono::high_resolution_clock::now();
        runSimulation(particles, root, 0.5, TIMESTEP);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_taken = end -  start;
        printf("Take: %i Time taken: %f ms \n", step + 1, time_taken.count() * 1000);   

        ++step;
    }

    root->clear();
    delete root;

    return 0;
}