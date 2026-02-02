/*
  Frenkel–Kontorova (FK) — extended effective potential traces (multi-seed)
  ------------------------------------------------------------------------

  This code matches the original logic/physics:
    - Twisted periodic boundary conditions (topological kink on a ring)
    - RK4 time stepping with damping
    - PN barrier EPN = E(unstable) - E(stable)
    - Hessian diagonalization (GSL) around relaxed stable configuration -> lambda_min, omega, effective mass M
    - 4 dynamical runs saved to 4 files, corresponding to different discrete "seeds" and kicks.

  Outputs (tab-separated columns):
    # t    X    EPN    M    Xdot    Xdot2

  Compile:
    gcc -O2 -Wall -Wextra -std=c11 -o potential_scan potential_effective_scan.c -lgsl -lgslcblas -lm
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LAMBDA 1.0
#define N 40
#define DT 0.1
#define STEPS 100000
#define GAMMA 1.0

/* Set to 1 to print debug info to stdout (as during development with your tutor) */
#define VERBOSE 0

typedef struct {
    double lambda;
    double omega;
} EigenData;

/* Twisted periodic difference (topological jump across boundary) */
static double delta_y(int i, int j, const double y[]) {
    double diff = y[i] - y[j];

    if (i == 0 && j == N - 1) {
        return diff + 2.0 * M_PI;
    } else if (i == N - 1 && j == 0) {
        return diff - 2.0 * M_PI;
    } else {
        return diff;
    }
}

/* One RK4 step. If a/b/c are 1, pin y[N/2], y[N/2-1], y[N/2+1] to pi after the step. */
static void evolve_system(double y[], double v[], int a, int b, int c) {
    double k1_y[N], k1_v[N], k2_y[N], k2_v[N], k3_y[N], k3_v[N], k4_y[N], k4_v[N];
    double y_tmp[N], v_tmp[N];

    /* k1 */
    for (int n = 0; n < N; n++) {
        int left  = (n - 1 + N) % N;
        int right = (n + 1) % N;
        k1_y[n] = DT * v[n];
        k1_v[n] = DT * (LAMBDA * (delta_y(right, n, y) + delta_y(left, n, y))
                        - sin(y[n]) - GAMMA * v[n]);
    }

    /* k2 */
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

    /* k3 */
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

    /* k4 */
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

    /* final update */
    for (int n = 0; n < N; n++) {
        y[n] += (1.0 / 6.0) * (k1_y[n] + 2.0*k2_y[n] + 2.0*k3_y[n] + k4_y[n]);
        v[n] += (1.0 / 6.0) * (k1_v[n] + 2.0*k2_v[n] + 2.0*k3_v[n] + k4_v[n]);
    }

    /* pinning (used for relaxation to converge to the saddle) */
    if (a == 1) {
        y[N/2] = M_PI;
        v[N/2] = 0.0;
    }
    if (b == 1) {
        y[N/2 - 1] = M_PI;
        v[N/2 - 1] = 0.0;
    }
    if (c == 1) {
        y[N/2 + 1] = M_PI;
        v[N/2 + 1] = 0.0;
    }
}

/* Energy difference EPN = E(unstable) - E(stable) */
static double Energia(const double ye[], const double yi[]) {
    double Energiae = 0.0;
    double Energiai = 0.0;

    for (int i = 0; i < N; i++) {
        if (i == N - 1) {
            Energiae += (1.0 - cos(ye[i])) + (LAMBDA/2.0) * pow(ye[0] + 2.0*M_PI - ye[i], 2);
            Energiai += (1.0 - cos(yi[i])) + (LAMBDA/2.0) * pow(yi[0] + 2.0*M_PI - yi[i], 2);
        } else {
            Energiae += (1.0 - cos(ye[i])) + (LAMBDA/2.0) * pow(ye[i+1] - ye[i], 2);
            Energiai += (1.0 - cos(yi[i])) + (LAMBDA/2.0) * pow(yi[i+1] - yi[i], 2);
        }
    }
    return (Energiai - Energiae);
}

/* Collective coordinates */
static void Colectivos(const double y[], const double v[], double M,
                       double *X, double *Xdot, double *Xdot2) {
    *X = (double)N;
    *Xdot = 0.0;
    *Xdot2 = 0.0;

    for (int i = 0; i < N; i++) {
        *X    += -y[i] / (2.0 * M_PI);
        *Xdot += -v[i] / (2.0 * M_PI);
        *Xdot2 += v[i] * v[i];
    }

    *Xdot2 = sqrt((*Xdot2) / M);
    *X = 0.5 + (*X);
}

/* Hessian matrix around ye[] */
static void build_hessian_twisted(gsl_matrix *H, const double ye[]) {
    gsl_matrix_set_zero(H);

    for (int j = 0; j < N; j++) {
        int jp1 = (j + 1) % N;
        double diag_val = cos(ye[j]) + 2.0 * LAMBDA;

        gsl_matrix_set(H, j, j, diag_val);
        gsl_matrix_set(H, j, jp1, -LAMBDA);
        gsl_matrix_set(H, jp1, j, -LAMBDA);
    }
}

int main(void) {
    if (VERBOSE) printf("Inicio del programa\n");

    double ye[N], ve[N], yi[N], vi[N];
    double EPN;

    /* ----------------------- RUN 1: center seed, kick - ----------------------- */

    /* Inicializamos (EXACTLY as your original) */
    for (int j = 0; j < N; j++) {
        if (j < N/2) {
            ye[j] = 0.0;
            yi[j] = 0.0;
        } else if (j == N/2) {
            ye[j] = 0.0;
            yi[j] = M_PI;
        } else {
            ye[j] = 2.0*M_PI;
            yi[j] = 2.0*M_PI;
        }
        ve[j] = 0.0;
        vi[j] = 0.0;
    }

    for (int t = 0; t < STEPS; t++) {
        evolve_system(ye, ve, 0, 0, 0);
        evolve_system(yi, vi, 1, 0, 0);
    }

    /* Open files (consistent naming: underscore) */
    FILE *fp  = fopen("Dibujo_del_potencial_2.txt", "w");
    FILE *fp2 = fopen("Dibujo_del_potencial_3.txt", "w");
    FILE *fp3 = fopen("Dibujo_del_potencial_4.txt", "w");
    FILE *fp4 = fopen("Dibujo_del_potencial_5.txt", "w");

    if (!fp || !fp2 || !fp3 || !fp4) {
        perror("No se pudo abrir uno de los archivos de salida");
        if (fp)  fclose(fp);
        if (fp2) fclose(fp2);
        if (fp3) fclose(fp3);
        if (fp4) fclose(fp4);
        return 1;
    }

    /* Headers */
    fprintf(fp,  "# t\tX\tEPN\tM\tXdot\tXdot2\n");
    fprintf(fp2, "# t\tX\tEPN\tM\tXdot\tXdot2\n");
    fprintf(fp3, "# t\tX\tEPN\tM\tXdot\tXdot2\n");
    fprintf(fp4, "# t\tX\tEPN\tM\tXdot\tXdot2\n");

    EPN = Energia(ye, yi);

    if (VERBOSE) {
        for (int i = 0; i < N; i++) {
            printf("ye[%d] = %.10f\n", i, ye[i]);
        }
    }

    /* Hessian diagonalization (same as original, but with full cleanup) */
    gsl_matrix *H = gsl_matrix_alloc(N, N);
    gsl_vector *eval = gsl_vector_alloc(N);
    gsl_matrix *evec = gsl_matrix_alloc(N, N);
    gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(N);

    build_hessian_twisted(H, ye);

    gsl_eigen_symmv(H, eval, evec, w);
    gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);

    EigenData eigenvalues[N];
    for (int i = 0; i < N; i++) {
        eigenvalues[i].lambda = gsl_vector_get(eval, i);
        eigenvalues[i].omega  = (eigenvalues[i].lambda > 1e-14) ? sqrt(eigenvalues[i].lambda) : 0.0;
    }

    double lambda_min = eigenvalues[0].lambda;
    double omega_soliton = eigenvalues[0].omega;
    double M = (omega_soliton > 1e-14) ? (0.5 * EPN / (omega_soliton * omega_soliton)) : 0.0;

    if (VERBOSE) {
        printf("lambda_min = %e\n", lambda_min);
        printf("EPN = %.10f\n", EPN);
        printf("M = %.10f\n", M);
    }

    /* Kick: exactly as original */
    vi[N/2] = -0.01;

    for (int t = 0; t < STEPS; t++) {
        evolve_system(yi, vi, 0, 0, 0);

        double X, Xdot, Xdot2;
        EPN = Energia(ye, yi);
        Colectivos(yi, vi, M, &X, &Xdot, &Xdot2);

        /* same sampling as original: write every step */
        fprintf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", t * DT, X, EPN, M, Xdot, Xdot2);
    }

    /* ----------------------- RUN 2: seed at N/2+1, kick + -------------------- */

    /* Inicializamos (EXACTLY as your original) */
    for (int j = 0; j < N; j++) {
        if (j < N/2) {
            ye[j] = 0.0;
            yi[j] = 0.0;
        } else if (j == N/2 + 1) {
            ye[j] = 0.0;
            yi[j] = M_PI;
        } else {
            ye[j] = 2.0*M_PI;
            yi[j] = 2.0*M_PI;
        }
        ve[j] = 0.0;
        vi[j] = 0.0;
    }

    for (int t = 0; t < STEPS; t++) {
        evolve_system(ye, ve, 0, 0, 0);
        evolve_system(yi, vi, 0, 0, 1);
    }

    /* Kick: exactly as original */
    vi[N/2 + 1] = +0.01;

    for (int t = 0; t < STEPS; t++) {
        evolve_system(yi, vi, 0, 0, 0);

        double X, Xdot, Xdot2;
        EPN = Energia(ye, yi);
        Colectivos(yi, vi, M, &X, &Xdot, &Xdot2);

        fprintf(fp2, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", t * DT, X, EPN, M, Xdot, Xdot2);
    }

    /* ----------------------- RUN 3: seed at N/2-1, kick - -------------------- */

    /* Inicializamos (EXACTLY as your original) */
    for (int j = 0; j < N; j++) {
        if (j < N/2) {
            ye[j] = 0.0;
            yi[j] = 0.0;
        } else if (j == N/2 - 1) {
            ye[j] = 0.0;
            yi[j] = M_PI;
        } else {
            ye[j] = 2.0*M_PI;
            yi[j] = 2.0*M_PI;
        }
        ve[j] = 0.0;
        vi[j] = 0.0;
    }

    for (int t = 0; t < STEPS; t++) {
        evolve_system(ye, ve, 0, 0, 0);
        evolve_system(yi, vi, 0, 1, 0);
    }

    /* Kick: exactly as original */
    vi[N/2 - 1] = -0.01;

    for (int t = 0; t < STEPS; t++) {
        evolve_system(yi, vi, 0, 0, 0);

        double X, Xdot, Xdot2;
        EPN = Energia(ye, yi);
        Colectivos(yi, vi, M, &X, &Xdot, &Xdot2);

        fprintf(fp3, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", t * DT, X, EPN, M, Xdot, Xdot2);
    }

    /* ----------------------- RUN 4: center seed, kick + ---------------------- */

    /* Inicializamos (EXACTLY as your original) */
    for (int j = 0; j < N; j++) {
        if (j < N/2) {
            ye[j] = 0.0;
            yi[j] = 0.0;
        } else if (j == N/2) {
            ye[j] = 0.0;
            yi[j] = M_PI;
        } else {
            ye[j] = 2.0*M_PI;
            yi[j] = 2.0*M_PI;
        }
        ve[j] = 0.0;
        vi[j] = 0.0;
    }

    for (int t = 0; t < STEPS; t++) {
        evolve_system(ye, ve, 0, 0, 0);
        evolve_system(yi, vi, 1, 0, 0);
    }

    /* Kick: exactly as original */
    vi[N/2] = +0.01;

    for (int t = 0; t < STEPS; t++) {
        evolve_system(yi, vi, 0, 0, 0);

        double X, Xdot, Xdot2;
        EPN = Energia(ye, yi);
        Colectivos(yi, vi, M, &X, &Xdot, &Xdot2);

        fprintf(fp4, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", t * DT, X, EPN, M, Xdot, Xdot2);
    }

    /* Cleanup */
    gsl_eigen_symmv_free(w);
    gsl_matrix_free(H);
    gsl_matrix_free(evec);
    gsl_vector_free(eval);

    fclose(fp);
    fclose(fp2);
    fclose(fp3);
    fclose(fp4);

    if (VERBOSE) printf("Fin.\n");
    return 0;
}
