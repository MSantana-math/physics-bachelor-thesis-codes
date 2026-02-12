/*
Frenkel–Kontorova model — effective mass estimate from collective coordinate

Idea:
  1) Relax an unstable (pinned) kink configuration with RK4 + damping.
  2) Apply a small velocity "kick" to one site.
  3) Track collective coordinate X and velocity Xdot.
  4) Estimate effective mass: M_eff = K / Xdot^2, where K = (1/2) sum_n v_n^2.

Outputs (tab-separated):
  - Masa_eff_t.txt   : kick at site N/2   with +0.1
  - Masa_eff_t2.txt  : kick at site N/2   with -0.1
  - Masa_eff_t3.txt  : kink centered at N/2+1, kick at N/2+1 with +0.1
  - Masa_eff_t4.txt  : kink centered at N/2-1, kick at N/2-1 with -0.1

Compile:
  gcc -O2 -std=c11 fk_effective_mass.c -lm -o fk_mass
Run:
  ./fk_mass
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
#define STEPS 10000
#define GAMMA 5.0

// Discrete difference with a 2p topological jump across the boundary (kink on a ring)
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

// One RK4 step for FK dynamics with damping.
// a,b,c: optionally pin y[N/2], y[N/2-1], y[N/2+1] to p (used during relaxation).
static void evolve_system(double y[], double v[], int a, int b, int c) {
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

    if (a == 1) y[N / 2] = M_PI;
    if (b == 1) y[N / 2 - 1] = M_PI;
    if (c == 1) y[N / 2 + 1] = M_PI;
}

// Kinetic energy: K = (1/2) sum_n v_n^2
static double kinetic_energy(const double v[]) {
    double K = 0.0;
    for (int n = 0; n < N; n++) {
        K += 0.5 * v[n] * v[n];
    }
    return K;
}

// Collective coordinate X and its velocity Xdot
static void colectivos(const double y[], const double v[], double *X, double *Xdot) {
    *X = (double)N;
    *Xdot = 0.0;

    for (int i = 0; i < N; i++) {
        *X    += -y[i] / (2.0 * M_PI);
        *Xdot += -v[i] / (2.0 * M_PI);
    }
    *X = 0.5 + *X;
}

// Effective mass estimate: M_eff = K / Xdot^2
static void medir_masa_efectiva(const double y[], const double v[],
                                double *M_eff, double *X, double *Xdot) {
    double K = kinetic_energy(v);
    colectivos(y, v, X, Xdot);

    if (fabs(*Xdot) < 1e-10) {
        *M_eff = 0.0;
    } else {
        *M_eff = K / ((*Xdot) * (*Xdot));
    }
}

static void write_header(FILE *f) {
    fprintf(f, "# t\tX\tM\tXdot\n");
}

int main(void) {
    double X, Xdot, M;
    double y[N], v[N];

    // --- CASE 1: center at N/2, relax with pin at N/2 ---
    for (int j = 0; j < N; j++) {
        if (j < N / 2) {
            y[j] = 0.0;
        } else if (j == N / 2) {
            y[j] = M_PI;
        } else {
            y[j] = 2.0 * M_PI;
        }
        v[j] = 0.0;
    }
    for (int t = 0; t < STEPS; t++) {
        evolve_system(y, v, 1, 0, 0);
    }

    FILE *fp  = fopen("Masa_eff_t.txt",  "w");
    FILE *fp2 = fopen("Masa_eff_t2.txt", "w");
    FILE *fp3 = fopen("Masa_eff_t3.txt", "w");
    FILE *fp4 = fopen("Masa_eff_t4.txt", "w");
    if (!fp || !fp2 || !fp3 || !fp4) {
        perror("No se pudo abrir uno de los archivos de salida");
        if (fp) fclose(fp);
        if (fp2) fclose(fp2);
        if (fp3) fclose(fp3);
        if (fp4) fclose(fp4);
        return 1;
    }

    write_header(fp);
    write_header(fp2);
    write_header(fp3);
    write_header(fp4);

    // Kick +0.1 at N/2
    for (int i = 0; i < N; i++) {
        if (i == N / 2) v[i] = 0.1;
    }
    for (int t = 0; t < STEPS; t++) {
        evolve_system(y, v, 0, 0, 0);
        medir_masa_efectiva(y, v, &M, &X, &Xdot);
        fprintf(fp, "%lf\t%lf\t%lf\t%lf\n", t * DT, X, M, Xdot);
    }

    // --- CASE 2: re-init same as case 1, relax, kick -0.1 at N/2 ---
    for (int j = 0; j < N; j++) {
        if (j < N / 2) {
            y[j] = 0.0;
        } else if (j == N / 2) {
            y[j] = M_PI;
        } else {
            y[j] = 2.0 * M_PI;
        }
        v[j] = 0.0;
    }
    for (int t = 0; t < STEPS; t++) {
        evolve_system(y, v, 1, 0, 0);
    }
    for (int i = 0; i < N; i++) {
        if (i == N / 2) v[i] = -0.1;
    }
    for (int t = 0; t < STEPS; t++) {
        evolve_system(y, v, 0, 0, 0);
        medir_masa_efectiva(y, v, &M, &X, &Xdot);
        fprintf(fp2, "%lf\t%lf\t%lf\t%lf\n", t * DT, X, M, Xdot);
    }

    // --- CASE 3: center at N/2+1 (keep EXACT init), relax with pin at N/2+1 ---
    for (int j = 0; j < N; j++) {
        if (j < N / 2 + 1) {
            y[j] = 0.0;
        } else if (j == N / 2 + 1) {
            y[j] = M_PI;
        } else {
            y[j] = 2.0 * M_PI;
        }
        v[j] = 0.0;
    }
    for (int t = 0; t < STEPS; t++) {
        evolve_system(y, v, 0, 0, 1);
    }
    for (int i = 0; i < N; i++) {
        if (i == N / 2 + 1) v[i] = +0.1;
    }
    for (int t = 0; t < STEPS; t++) {
        evolve_system(y, v, 0, 0, 0);
        medir_masa_efectiva(y, v, &M, &X, &Xdot);
        fprintf(fp3, "%lf\t%lf\t%lf\t%lf\n", t * DT, X, M, Xdot);
    }

    // --- CASE 4: center at N/2-1 (keep EXACT init), relax with pin at N/2-1 ---
    for (int j = 0; j < N; j++) {
        if (j < N / 2 - 1) {
            y[j] = 0.0;
        } else if (j == N / 2 - 1) {
            y[j] = M_PI;
        } else {
            y[j] = 2.0 * M_PI;
        }
        v[j] = 0.0;
    }
    for (int t = 0; t < STEPS; t++) {
        evolve_system(y, v, 0, 1, 0);
    }
    for (int i = 0; i < N; i++) {
        if (i == N / 2 - 1) v[i] = -0.1;
    }
    for (int t = 0; t < STEPS; t++) {
        evolve_system(y, v, 0, 0, 0);
        medir_masa_efectiva(y, v, &M, &X, &Xdot);
        fprintf(fp4, "%lf\t%lf\t%lf\t%lf\n", t * DT, X, M, Xdot);
    }

    fclose(fp);
    fclose(fp2);
    fclose(fp3);
    fclose(fp4);
    return 0;
}
