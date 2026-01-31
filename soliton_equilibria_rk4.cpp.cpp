/*
Frenkel–Kontorova model — soliton equilibrium profiles (stable & unstable)
Method: RK4 time integration + damping (relaxation to equilibrium)

Outputs:
  - perfiles_soliton.csv  (columns: i, u_stable, u_unstable)

Compile:
  gcc -O2 -std=c11 soliton_equilibria_rk4.c -lm -o soliton_eq
Run:
  ./soliton_eq
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LAMBDA 1.0
#define N 40
#define DT 0.1
#define STEPS 100000
#define GAMMA 1.0

// Discrete difference with a 2p topological jump across the boundary (kink on a ring)
double delta_y(int i, int j, const double y[]) {
    double diff = y[i] - y[j];

    if (i == 0 && j == N - 1) {
        return diff + 2.0 * M_PI;
    } else if (i == N - 1 && j == 0) {
        return diff - 2.0 * M_PI;
    } else {
        return diff;
    }
}

// One RK4 step for the FK dynamics with damping.
// If pin_center == 1, enforce y[N/2] = pi (unstable equilibrium / saddle configuration).
void rk4_step(double y[], double v[], int pin_center) {
    double k1_y[N], k1_v[N], k2_y[N], k2_v[N], k3_y[N], k3_v[N], k4_y[N], k4_v[N];
    double y_tmp[N], v_tmp[N];

    // k1
    for (int n = 0; n < N; n++) {
        int left  = (n - 1 + N) % N;
        int right = (n + 1) % N;
        k1_y[n] = DT * v[n];
        k1_v[n] = DT * (LAMBDA * (delta_y(right, n, y) + delta_y(left, n, y))
                        - sin(y[n]) - GAMMA * v[n]);
    }

    // k2
    for (int n = 0; n < N; n++) {
        y_tmp[n] = y[n] + 0.5 * k1_y[n];
        v_tmp[n] = v[n] + 0.5 * k1_v[n];
    }
    for (int n = 0; n < N; n++) {
        int left  = (n - 1 + N) % N;
        int right = (n + 1) % N;
        k2_y[n] = DT * v_tmp[n];
        k2_v[n] = DT * (LAMBDA * (delta_y(right, n, y_tmp) + delta_y(left, n, y_tmp))
                        - sin(y_tmp[n]) - GAMMA * v_tmp[n]);
    }

    // k3
    for (int n = 0; n < N; n++) {
        y_tmp[n] = y[n] + 0.5 * k2_y[n];
        v_tmp[n] = v[n] + 0.5 * k2_v[n];
    }
    for (int n = 0; n < N; n++) {
        int left  = (n - 1 + N) % N;
        int right = (n + 1) % N;
        k3_y[n] = DT * v_tmp[n];
        k3_v[n] = DT * (LAMBDA * (delta_y(right, n, y_tmp) + delta_y(left, n, y_tmp))
                        - sin(y_tmp[n]) - GAMMA * v_tmp[n]);
    }

    // k4
    for (int n = 0; n < N; n++) {
        y_tmp[n] = y[n] + k3_y[n];
        v_tmp[n] = v[n] + k3_v[n];
    }
    for (int n = 0; n < N; n++) {
        int left  = (n - 1 + N) % N;
        int right = (n + 1) % N;
        k4_y[n] = DT * v_tmp[n];
        k4_v[n] = DT * (LAMBDA * (delta_y(right, n, y_tmp) + delta_y(left, n, y_tmp))
                        - sin(y_tmp[n]) - GAMMA * v_tmp[n]);
    }

    // Final update
    for (int n = 0; n < N; n++) {
        y[n] += (1.0 / 6.0) * (k1_y[n] + 2.0*k2_y[n] + 2.0*k3_y[n] + k4_y[n]);
        v[n] += (1.0 / 6.0) * (k1_v[n] + 2.0*k2_v[n] + 2.0*k3_v[n] + k4_v[n]);
    }

    if (pin_center) {
        y[N / 2] = M_PI;
        v[N / 2] = 0.0;  // keep pinned point consistent (optional but clean)
    }
}

int main() {
    double u_stable[N], v_stable[N], u_unstable[N], v_unstable[N];

    // Initialization (step-like profile)
    for (int j = 0; j < N; j++) {
        if (j < N / 2) {
            u_stable[j]   = 0.0;
            u_unstable[j] = 0.0;
        } else if (j == N / 2) {
            u_stable[j]   = 0.0;
            u_unstable[j] = M_PI;  // center pinned to pi for unstable configuration
        } else {
            u_stable[j]   = 2.0 * M_PI;
            u_unstable[j] = 2.0 * M_PI;
        }
        v_stable[j]   = 0.0;
        v_unstable[j] = 0.0;
    }

    // Relaxation loop
    for (int t = 0; t < STEPS; t++) {
        rk4_step(u_stable,   v_stable,   0);
        rk4_step(u_unstable, v_unstable, 1);
    }

    // Output CSV
    FILE *f = fopen("perfiles_soliton.csv", "w");
    if (f == NULL) {
        printf("Error al abrir el archivo.\n");
        return 1;
    }

    fprintf(f, "i,u_stable,u_unstable\n");
    for (int n = 0; n < N; n++) {
        fprintf(f, "%d,%.15f,%.15f\n", n, u_stable[n], u_unstable[n]);
    }

    fclose(f);
    printf("Archivo 'perfiles_soliton.csv' guardado correctamente.\n");
    return 0;
}
