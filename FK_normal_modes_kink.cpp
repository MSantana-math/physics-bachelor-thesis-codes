/*
Frenkel–Kontorova model — normal modes around a topological kink (ring geometry)
Method:
  1) RK4 time integration + damping (relaxation to equilibrium)
  2) Build Hessian at the relaxed configuration
  3) Diagonalize Hessian with GSL to obtain eigenvalues and frequencies

Output:
  - modos_con_fuerzaN8.txt   (columns: mode_index, lambda, omega)

Compile (example):
  gcc -O2 -std=c11 fk_normal_modes_kink_gsl.c -lm -lgsl -lgslcblas -o fk_modes

Run:
  ./fk_modes
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_eigen.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N 8
#define DT 0.01
#define STEPS 100000
#define LAMBDA 1.0
#define GAMMA 1.0
#define F 0.0  // External force (set != 0 to tilt / drive)

static double delta_y(int i, int j, const double y[]) {
    double diff = y[i] - y[j];
    if (i == 0 && j == N - 1) diff += 2.0 * M_PI;
    else if (i == N - 1 && j == 0) diff -= 2.0 * M_PI;
    return diff;
}

// One RK4 step for the FK dynamics with topological jump across the boundary.
// If pin_center == 1, enforce y[N/2] = pi (optional constraint).
static void rk4_step_topological(double y[], double v[], int pin_center) {
    double k1_y[N], k1_v[N], k2_y[N], k2_v[N], k3_y[N], k3_v[N], k4_y[N], k4_v[N];
    double y_tmp[N], v_tmp[N];

    // k1
    for (int n = 0; n < N; n++) {
        int left  = (n - 1 + N) % N;
        int right = (n + 1) % N;
        k1_y[n] = DT * v[n];
        k1_v[n] = DT * (LAMBDA * (delta_y(right, n, y) + delta_y(left, n, y))
                        - sin(y[n]) - GAMMA * v[n] + F);
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
                        - sin(y_tmp[n]) - GAMMA * v_tmp[n] + F);
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
                        - sin(y_tmp[n]) - GAMMA * v_tmp[n] + F);
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
                        - sin(y_tmp[n]) - GAMMA * v_tmp[n] + F);
    }

    // Update
    for (int n = 0; n < N; n++) {
        y[n] += (k1_y[n] + 2.0*k2_y[n] + 2.0*k3_y[n] + k4_y[n]) / 6.0;
        v[n] += (k1_v[n] + 2.0*k2_v[n] + 2.0*k3_v[n] + k4_v[n]) / 6.0;
    }

    if (pin_center) {
        y[N / 2] = M_PI;
    }
}

// Build Hessian matrix H_ij = d^2 V / dy_i dy_j evaluated at y[]
static void build_hessian(gsl_matrix *H, const double y[]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double value = 0.0;

            if (i == j) {
                value = 2.0 * LAMBDA + cos(y[i]);
            } else if (j == i + 1 || j == i - 1 ||
                       (i == 0 && j == N - 1) || (i == N - 1 && j == 0)) {
                value = -LAMBDA;
            }

            gsl_matrix_set(H, i, j, value);
        }
    }
}

int main(void) {
    double y[N], v[N];

    // Step-like kink initialization on a ring
    for (int i = 0; i < N; i++) {
        if (i < N / 2) y[i] = 0.0;
        else          y[i] = 2.0 * M_PI;
        v[i] = 0.0;
    }

    // Relaxation
    for (int t = 0; t < STEPS; t++) {
        rk4_step_topological(y, v, 0);
    }

    // Hessian + eigendecomposition
    gsl_matrix *H = gsl_matrix_alloc(N, N);
    build_hessian(H, y);

    gsl_vector *eval = gsl_vector_alloc(N);
    gsl_matrix *evec = gsl_matrix_alloc(N, N);
    gsl_eigen_symmv_workspace *ws = gsl_eigen_symmv_alloc(N);

    gsl_eigen_symmv(H, eval, evec, ws);
    gsl_eigen_symmv_free(ws);

    // Output
    FILE *f = fopen("modos_con_fuerzaN8.txt", "w");
    if (!f) {
        perror("No se pudo abrir el archivo de salida");
        gsl_matrix_free(H);
        gsl_vector_free(eval);
        gsl_matrix_free(evec);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        double lambda = gsl_vector_get(eval, i);
        double omega  = (lambda > 0.0) ? sqrt(lambda) : 0.0;
        fprintf(f, "%d\t%.10f\t%.10f\n", i, lambda, omega);
    }

    fclose(f);
    gsl_matrix_free(H);
    gsl_vector_free(eval);
    gsl_matrix_free(evec);

    printf("¡Cálculo completado! Resultados en 'modos_con_fuerzaN8.txt'\n");
    return 0;
}
