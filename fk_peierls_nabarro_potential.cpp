/*
  Frenkel–Kontorova (FK) — Peierls–Nabarro barrier and "effective potential" trace
  ------------------------------------------------------------------------------

  What this code does (same as your original):
    1) Relaxes two FK configurations with damping:
         - Stable minimum (all sites at 0 or 2p)
         - Unstable "saddle" (one site pinned at p)
       using RK4 time stepping of the damped equations of motion.

    2) Computes the Peierls–Nabarro barrier:
         EPN = E(unstable) - E(stable)

    3) Builds the Hessian around the relaxed stable configuration ye[] and
       diagonalizes it with GSL to obtain the smallest eigenvalue and frequency,
       used to estimate an effective collective mass M.

    4) Runs two dynamical trajectories (positive and negative kick) for the
       unstable configuration (now free, not pinned), saving:
         t, X, EPN, M, Xdot, Xdot2
       into: Dibujo_del_potencial.txt

  Dependencies:
    - GSL (GNU Scientific Library)

  Compile:
    gcc -O2 -Wall -Wextra -o potential potential_effective.c -lgsl -lgslcblas -lm

  Output:
    Dibujo_del_potencial.txt

  Notes:
    - This file keeps the same physics/logic of the original code; changes are
      only readability + safe printing (avoid out-of-bounds debug print).
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>

/* ------------------------- Simulation parameters -------------------------- */

/* FK coupling (often denoted lambda in your notes/TFG) */
#define LAMBDA 0.01

/* Number of sites (ring) */
#define N 8

/* Time step for RK4 */
#define DT 0.1

/* Number of integration steps used for relaxation and for trajectories */
#define STEPS 100000

/* Damping (gamma). Large gamma => strong relaxation. */
#define GAMMA 1.0

/* ----------------------------- Small structs ------------------------------ */

typedef struct{
    double lambda;  /* eigenvalue of Hessian */
    double omega;   /* frequency: sqrt(lambda) if lambda>0 */
} EigenData;

/* ------------------------- Twisted periodic difference -------------------- */
/*
   delta_y(i,j,y) = y[i]-y[j], but with a 2p "twist" across the boundary
   (i=0,j=N-1 and i=N-1,j=0). This encodes the topological kink on a ring.
*/
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

/* ------------------------- One RK4 step of dynamics ----------------------- */
/*
  Integrates one RK4 step of:
      ydot = v
      vdot = LAMBDA*(?y_right + ?y_left) - sin(y) - GAMMA*v

  Parameter a:
    - if a==1, we pin y[N/2]=p after each step (to converge to the unstable saddle).
    - if a==0, no pinning (normal dynamics).
*/
static void evolve_system(double y[], double v[], int a) {

    double k1_y[N], k1_v[N];
    double k2_y[N], k2_v[N];
    double k3_y[N], k3_v[N];
    double k4_y[N], k4_v[N];

    double y_tmp[N], v_tmp[N];

    /* k1 */
    for(int n = 0; n < N; n++){
        int left  = (n - 1 + N) % N;
        int right = (n + 1) % N;

        k1_y[n] = DT * v[n];
        k1_v[n] = DT * ( LAMBDA*( delta_y(right,n,y) + delta_y(left,n,y) )
                         - sin(y[n]) - GAMMA*v[n] );
    }

    /* k2 */
    for(int n = 0; n < N; n++){
        y_tmp[n] = y[n] + 0.5 * k1_y[n];
        v_tmp[n] = v[n] + 0.5 * k1_v[n];
    }
    for(int n = 0; n < N; n++){
        int left  = (n - 1 + N) % N;
        int right = (n + 1) % N;

        k2_y[n] = DT * v_tmp[n];
        k2_v[n] = DT * ( LAMBDA*( delta_y(right,n,y_tmp) + delta_y(left,n,y_tmp) )
                         - sin(y_tmp[n]) - GAMMA*v_tmp[n] );
    }

    /* k3 */
    for(int n = 0; n < N; n++){
        y_tmp[n] = y[n] + 0.5 * k2_y[n];
        v_tmp[n] = v[n] + 0.5 * k2_v[n];
    }
    for(int n = 0; n < N; n++){
        int left  = (n - 1 + N) % N;
        int right = (n + 1) % N;

        k3_y[n] = DT * v_tmp[n];
        k3_v[n] = DT * ( LAMBDA*( delta_y(right,n,y_tmp) + delta_y(left,n,y_tmp) )
                         - sin(y_tmp[n]) - GAMMA*v_tmp[n] );
    }

    /* k4 */
    for(int n = 0; n < N; n++){
        y_tmp[n] = y[n] + k3_y[n];
        v_tmp[n] = v[n] + k3_v[n];
    }
    for(int n = 0; n < N; n++){
        int left  = (n - 1 + N) % N;
        int right = (n + 1) % N;

        k4_y[n] = DT * v_tmp[n];
        k4_v[n] = DT * ( LAMBDA*( delta_y(right,n,y_tmp) + delta_y(left,n,y_tmp) )
                         - sin(y_tmp[n]) - GAMMA*v_tmp[n] );
    }

    /* final update */
    for(int n = 0; n < N; n++){
        y[n] += (1.0/6.0) * (k1_y[n] + 2.0*k2_y[n] + 2.0*k3_y[n] + k4_y[n]);
        v[n] += (1.0/6.0) * (k1_v[n] + 2.0*k2_v[n] + 2.0*k3_v[n] + k4_v[n]);
    }

    /* pinning for the unstable equilibrium */
    if(a == 1){
        y[N/2] = M_PI;
    }
}

/* ------------------------------ Energies --------------------------------- */
/*
  FK energy:
     E = S_i [ (1 - cos(y_i)) + (LAMBDA/2)*(y_{i+1}-y_i)^2 ]
  with twisted periodic boundary on the last link:
     (y_0 + 2p - y_{N-1})
  Returns:
     EPN = E(unstable) - E(stable)
*/
static double Energia(const double ye[], const double yi[]){
    double Energiae = 0.0; /* stable */
    double Energiai = 0.0; /* unstable */

    for(int i = 0; i < N; i++){
        if(i == N-1){
            Energiae += (1.0 - cos(ye[i])) + (LAMBDA/2.0)*pow(ye[0] + 2.0*M_PI - ye[i], 2);
            Energiai += (1.0 - cos(yi[i])) + (LAMBDA/2.0)*pow(yi[0] + 2.0*M_PI - yi[i], 2);
        } else {
            Energiae += (1.0 - cos(ye[i])) + (LAMBDA/2.0)*pow(ye[i+1] - ye[i], 2);
            Energiai += (1.0 - cos(yi[i])) + (LAMBDA/2.0)*pow(yi[i+1] - yi[i], 2);
        }
    }
    return (Energiai - Energiae);
}

/* ------------------------- Collective coordinates ------------------------- */
/*
  Defines:
     X    = 0.5 + N - S_i y_i/(2p)
     Xdot =      - S_i v_i/(2p)
     Xdot2 = sqrt( (S_i v_i^2)/M )   (a speed-like scalar)
*/
static void Colectivos(const double y[], const double v[], double M,
                       double *X, double *Xdot, double *Xdot2){

    *X     = (double)N;
    *Xdot  = 0.0;
    *Xdot2 = 0.0;

    for(int i = 0; i < N; i++){
        *X    += -y[i]/(2.0*M_PI);
        *Xdot += -v[i]/(2.0*M_PI);
        *Xdot2 += v[i]*v[i];
    }

    *Xdot2 = sqrt((*Xdot2) / M);
    *X     = 0.5 + (*X);
}

/* ----------------------------- Hessian (GSL) ------------------------------ */
/*
  Hessian around configuration ye[]:
     H_ii = cos(ye_i) + 2*LAMBDA
     H_{i,i±1} = -LAMBDA   (with periodic indexing)
*/
static void build_hessian_twisted(gsl_matrix *H, const double ye[]) {

    gsl_matrix_set_zero(H);

    for(int j = 0; j < N; j++){
        int jp1 = (j + 1) % N;
        double diag_val = cos(ye[j]) + 2.0 * LAMBDA;

        gsl_matrix_set(H, j, j,   diag_val);
        gsl_matrix_set(H, j, jp1, -LAMBDA);
        gsl_matrix_set(H, jp1, j, -LAMBDA);
    }
}

/* ----------------------------- Initialization ----------------------------- */
/*
  Initial guess (same as your original):
    - Left half: 0
    - Center: ye[N/2]=0 (stable) / yi[N/2]=p (unstable)
    - Right half: 2p
    - velocities = 0
*/
static void init_configs(double ye[], double ve[], double yi[], double vi[]){

    for(int j = 0; j < N; j++){

        if(j < N/2){
            ye[j] = 0.0;
            yi[j] = 0.0;
        }
        else if(j == N/2){
            ye[j] = 0.0;
            yi[j] = M_PI;
        }
        else{
            ye[j] = 2.0*M_PI;
            yi[j] = 2.0*M_PI;
        }

        ve[j] = 0.0;
        vi[j] = 0.0;
    }
}

/* ---------------------------------- main ---------------------------------- */

int main(void){

    printf("Inicio del programa\n");

    double ye[N], ve[N];
    double yi[N], vi[N];

    double EPN;

    /* 1) Initialize and relax stable & unstable equilibria */
    init_configs(ye, ve, yi, vi);

    for(int t = 0; t < STEPS; t++){
        evolve_system(ye, ve, 0); /* stable relax */
        evolve_system(yi, vi, 1); /* unstable relax (pinned at p) */
    }

    printf("Abriendo archivo de salida...\n");
    FILE *fp = fopen("Dibujo_del_potencial.txt", "w");
    if(!fp){
        perror("No se pudo abrir el archivo");
        return 1;
    }

    /* 2) Compute PN barrier */
    printf("Calculando EPN...\n");
    EPN = Energia(ye, yi);

    fprintf(fp, "# t\tX\tEPN\tM\tXdot\tXdot2\n");

    /* Safe debug print (N entries, no out-of-bounds) */
    for(int i = 0; i < N; i++){
        printf("ye[%d] = %.10f\n", i, ye[i]);
    }

    /* 3) Hessian diagonalization around relaxed stable configuration */
    printf("Diagonalizando Hessiana...\n");

    gsl_matrix *H = gsl_matrix_alloc(N, N);
    build_hessian_twisted(H, ye);

    gsl_vector *eval = gsl_vector_alloc(N);
    gsl_matrix *evec = gsl_matrix_alloc(N, N);
    gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(N);

    gsl_eigen_symmv(H, eval, evec, w);
    gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);

    EigenData eigenvalues[N];
    for(int i = 0; i < N; i++){
        eigenvalues[i].lambda = gsl_vector_get(eval, i);
        eigenvalues[i].omega  = (eigenvalues[i].lambda > 1e-14) ? sqrt(eigenvalues[i].lambda) : 0.0;
    }

    double lambda_min   = eigenvalues[0].lambda;
    double omega_soliton = eigenvalues[0].omega;

    /* Effective mass estimate (as in your code) */
    double M = (omega_soliton > 1e-14) ? (0.5 * EPN / (omega_soliton * omega_soliton)) : 0.0;

    printf("Hessiana diagonalizada\n");
    printf("lambda_min = %e\n", lambda_min);
    printf("EPN = %.10f\n", EPN);
    printf("omega_soliton = %.10e\n", omega_soliton);
    printf("M = %.10f\n", M);

    /* 4) Trajectory 1: positive kick on the center site */
    for(int i = 0; i < N; i++){
        if(i == N/2) vi[i] = 0.01;
    }

    for(int t = 0; t < STEPS; t++){
        evolve_system(yi, vi, 0);

        double X, Xdot, Xdot2;
        EPN = Energia(ye, yi);
        Colectivos(yi, vi, M, &X, &Xdot, &Xdot2);

        /* same sampling as original (t%1==0 writes every step) */
        if(t % 1 == 0){
            fprintf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", t * DT, X, EPN, M, Xdot, Xdot2);
        }
    }

    /* 5) Re-initialize and relax again (same as your original flow) */
    init_configs(ye, ve, yi, vi);

    for(int t = 0; t < STEPS; t++){
        evolve_system(ye, ve, 0);
        evolve_system(yi, vi, 1);
    }

    /* Trajectory 2: negative kick */
    for(int i = 0; i < N; i++){
        if(i == N/2) vi[i] = -0.01;
    }

    for(int t = 0; t < STEPS; t++){
        evolve_system(yi, vi, 0);

        double X, Xdot, Xdot2;
        EPN = Energia(ye, yi);
        Colectivos(yi, vi, M, &X, &Xdot, &Xdot2);

        if(t % 1 == 0){
            fprintf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", t * DT, X, EPN, M, Xdot, Xdot2);
        }
    }

    /* Cleanup */
    gsl_eigen_symmv_free(w);
    gsl_matrix_free(H);
    gsl_matrix_free(evec);
    gsl_vector_free(eval);

    fclose(fp);

    printf("Fin. Datos guardados en Dibujo_del_potencial.txt\n");
    return 0;
}
