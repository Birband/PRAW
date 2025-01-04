#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <array>
#include <cstdlib> // rand
#include <chrono>
#include <iostream>

// Simulation parameters
const double G = 6.67e-11;
const double TIMESTEP = 0.1;
const int NUM_STEPS = 1000;
const int NUMBER_OF_PARTICLES = 100000;

// Particle generation parameters
const double MAX_RADIUS = 5e5;
const double MIN_RADIUS = 5e4;
const double MAX_MASS = 1e20;
const double MIN_MASS = 1e15;
const double CENTRAL_MASS = 1e24;

// Rendering parameters
const double SCALE = 4.2e-4;
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
            double epsilon = 0.5;
            double r = std::sqrt(dx * dx + dy * dy + epsilon * epsilon); // Avoid division by zero
            double force = (G * particle->mass * root->particles[0]->mass) / ((r * r * r  + (epsilon * epsilon)));
            particle->fx += force * dx / r;
            particle->fy += force * dy / r;
        }
    } else {
        double size = std::max(root->x_max - root->x_min, root->y_max - root->y_min);
        double dx = root->center_x - particle->x;
        double dy = root->center_y - particle->y;
        double epsilon = 0.5;
        double r = std::sqrt(dx * dx + dy * dy + epsilon * epsilon); // Avoid division by zero

        if (size / r < theta) {
            double force = (G * particle->mass * root->particles[0]->mass) / ((r * r  + (epsilon * epsilon)));
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
    // return std::min(1.0, (1.0 + log10(mass * SCALE))); 
    if (mass > 1e20) return 10;
    else return 1.0;
}

void generateParticles(std::vector<Particle>& particles, int num_particles) {
    // Define positions of two central masses along the x-axis
    double mass1_x = -MAX_RADIUS;  // Central mass 1 position
    double mass2_x = MAX_RADIUS;   // Central mass 2 position

    // Add the two central masses to the particle list
    particles.push_back(Particle(CENTRAL_MASS, mass1_x, 0, 0, 0)); // First central mass
    particles.push_back(Particle(CENTRAL_MASS, mass2_x, 0, 0, 0)); // First central mass

    for (int i = 0; i < num_particles; ++i) {
        double center_x = i % 2 == 0 ? mass1_x : mass2_x;

        double theta = 2 * M_PI * rand() / RAND_MAX;
        double r = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * (rand() / double(RAND_MAX));
        double x = r * std::cos(theta);
        double y = r * std::sin(theta);

        double mass = MIN_MASS + (MAX_MASS - MIN_MASS) * (rand() / double(RAND_MAX));

        double velocity_scaling = 1.0; 

        double orbital_velocity = std::sqrt(G * CENTRAL_MASS / r) * velocity_scaling;

        double vx = -y * orbital_velocity / r;
        double vy = x * orbital_velocity / r;

        // Add the particle to the list
        particles.push_back(Particle(mass, x + center_x, y, vx, vy));
    }
}

void runSimulation(std::vector<Particle>& particles, Node* root, double theta, double dt) {
    root->clear();
    for (Particle& p : particles) {
        p.fx = 0;
        p.fy = 0;
    }
    
    for (Particle& p : particles) {
        insertParticle(root, &p);
    }

    for (Particle& p : particles) {
        calculateForce(&p, root, theta);
    }

    for (Particle& p : particles) {
        updatePosition(&p, dt);
    }
}

void drawParticles(const std::vector<Particle>& particles, sf::RenderWindow& window) {
    for (const Particle& p : particles) {
        double sizeBasedOnMass = calculateParticleSize(p.mass);
        sf::CircleShape particleShape(sizeBasedOnMass);
        particleShape.setFillColor(sf::Color::White);

        double x = WINDOW_SIZE / 2 + p.x * SCALE;
        double y = WINDOW_SIZE / 2 + p.y * SCALE;

        if (x >= 0 && x < WINDOW_SIZE && y >= 0 && y < WINDOW_SIZE) {
            particleShape.setPosition(x, y);
            particleShape.setOrigin(sizeBasedOnMass, sizeBasedOnMass);
            window.draw(particleShape);
        }
    }
}

void displayCountSpeed(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point stop, sf::RenderWindow& window, sf::Font& font) {
    std::chrono::duration<double> elapsed = stop - start;
    double count = elapsed.count() * 1000;
    double fps = 1000 / count;
    std::string text = "Frame time: " + std::to_string(count) + "ms\n"
                     + "FPS: " + std::to_string(fps);
    sf::Text fpsText(text, font, 20);
    window.draw(fpsText);
}


int main() {
    sf::Font font;
    if (!font.loadFromFile("Arial.ttf")) {
        return -1;
    }

    std::vector<Particle> particles;
    generateParticles(particles, NUMBER_OF_PARTICLES);

    Node* root = new Node(-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS);

    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Body Simulation");
    window.setFramerateLimit(60);

    int step = 0;
    while (window.isOpen() && step < NUM_STEPS) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        runSimulation(particles, root, 0.5, TIMESTEP);
        auto end = std::chrono::high_resolution_clock::now();

        window.clear();
        drawParticles(particles, window);
        displayCountSpeed(start, end, window, font);
        window.display();
    
        ++step;
    }

    root->clear();
    delete root;

    return 0;
}
