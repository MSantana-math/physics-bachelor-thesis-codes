/*
Frenkel–Kontorova model — Peierls–Nabarro barrier vs coupling (EPN as a function of Lambda)
Method: RK4 time integration + damping (relaxation to equilibrium)

Idea:
  For each Lambda in [Lambdamin, Lambdamax], relax:
    - a stable kink profile (fixed endpoints)
    - an unstable/saddle kink profile (pinned center at pi)
  Then compute EPN = E_unstable - E_stable.

Output:
  - EPN_LAMBDA.txt   (columns: Lambda, EPN)

Compile:
  gcc -O2 -std=c11 epn_vs_lambda.c -lm -o epn_vs_lambda
Run:
  ./epn_vs_lambda
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NN 40
#define DT 0.01          // Paso de tiempo
#define STEPS 10000      // Número de pasos de simulación
#define V_INICIAL 0.0    // Velocidad inicial del solitón (no usada, se mantiene)
#define GAMMA 1.0
#define A 10
#define Lambdamin 0.0
#define Lambdamax 2.0

// Función de evolución Runge–Kutta (estable)
void evolve_systemestable(double ye[], double ve[], double LAMBDA) {
    double k1_y[NN], k1_v[NN], k2_y[NN], k2_v[NN], k3_y[NN], k3_v[NN], k4_y[NN], k4_v[NN];

    for (int t = 0; t < STEPS; t++) {
        for (int n = 1; n < NN - 1; n++) { // Excluimos extremos
            // Condiciones de contorno periódicas (vecinos)
            int left  = (n - 1 + NN) % NN;
            int right = (n + 1) % NN;

            // k1
            k1_y[n] = DT * ve[n];
            k1_v[n] = DT * (LAMBDA * (ye[right] + ye[left] - 2.0 * ye[n]) - sin(ye[n]) - GAMMA * ve[n]);

            // k2
            k2_y[n] = DT * (ve[n] + 0.5 * k1_v[n]);
            k2_v[n] = DT * (LAMBDA * (ye[right] + ye[left] - 2.0 * (ye[n] + 0.5 * k1_y[n]))
                            - sin(ye[n] + 0.5 * k1_y[n])
                            - GAMMA * (ve[n] + 0.5 * k1_v[n]));

            // k3
            k3_y[n] = DT * (ve[n] + 0.5 * k2_v[n]);
            k3_v[n] = DT * (LAMBDA * (ye[right] + ye[left] - 2.0 * (ye[n] + 0.5 * k2_y[n]))
                            - sin(ye[n] + 0.5 * k2_y[n])
                            - GAMMA * (ve[n] + 0.5 * k2_v[n]));

            // k4
            k4_y[n] = DT * (ve[n] + k3_v[n]);
            k4_v[n] = DT * (LAMBDA * (ye[right] + ye[left] - 2.0 * (ye[n] + k3_y[n]))
                            - sin(ye[n] + k3_y[n])
                            - GAMMA * (ve[n] + k3_v[n]));
        }

        // Actualizar posición y velocidad
        for (int n = 1; n < NN - 1; n++) {
            ye[n] += (1.0 / 6.0) * (k1_y[n] + 2.0 * k2_y[n] + 2.0 * k3_y[n] + k4_y[n]);
            ve[n] += (1.0 / 6.0) * (k1_v[n] + 2.0 * k2_v[n] + 2.0 * k3_v[n] + k4_v[n]);
        }

        // Reforzar condiciones de contorno
        ye[0]      = 0.0;
        ye[NN - 1] = 2.0 * M_PI;
    }
}

// Runge–Kutta (solitón inestable / saddle): pin en el centro
void evolve_systeminestable(double yi[], double vi[], double LAMBDA) {
    double k1_y[NN], k1_v[NN], k2_y[NN], k2_v[NN], k3_y[NN], k3_v[NN], k4_y[NN], k4_v[NN];

    for (int t = 0; t < STEPS; t++) {
        for (int n = 1; n < NN - 1; n++) { // Excluimos extremos
            if (n == NN / 2) continue;

            int left  = (n - 1 + NN) % NN;
            int right = (n + 1) % NN;

            // k1
            k1_y[n] = DT * vi[n];
            k1_v[n] = DT * (LAMBDA * (yi[right] + yi[left] - 2.0 * yi[n]) - sin(yi[n]) - GAMMA * vi[n]);

            // k2
            k2_y[n] = DT * (vi[n] + 0.5 * k1_v[n]);
            k2_v[n] = DT * (LAMBDA * (yi[right] + yi[left] - 2.0 * (yi[n] + 0.5 * k1_y[n]))
                            - sin(yi[n] + 0.5 * k1_y[n])
                            - GAMMA * (vi[n] + 0.5 * k1_v[n]));

            // k3
            k3_y[n] = DT * (vi[n] + 0.5 * k2_v[n]);
            k3_v[n] = DT * (LAMBDA * (yi[right] + yi[left] - 2.0 * (yi[n] + 0.5 * k2_y[n]))
                            - sin(yi[n] + 0.5 * k2_y[n])
                            - GAMMA * (vi[n] + 0.5 * k2_v[n]));

            // k4
            k4_y[n] = DT * (vi[n] + k3_v[n]);
            k4_v[n] = DT * (LAMBDA * (yi[right] + yi[left] - 2.0 * (yi[n] + k3_y[n]))
                            - sin(yi[n] + k3_y[n])
                            - GAMMA * (vi[n] + k3_v[n]));
        }

        for (int n = 1; n < NN - 1; n++) {
            if (n == NN / 2) continue;

            yi[n] += (1.0 / 6.0) * (k1_y[n] + 2.0 * k2_y[n] + 2.0 * k3_y[n] + k4_y[n]);
            vi[n] += (1.0 / 6.0) * (k1_v[n] + 2.0 * k2_v[n] + 2.0 * k3_v[n] + k4_v[n]);
        }

        // Reforzar condiciones de contorno + pin del centro
        yi[0]      = 0.0;
        yi[NN - 1] = 2.0 * M_PI;
        yi[NN / 2] = M_PI;
    }
}

double Energia(double ye[], double yi[], double LAMBDA) {
    double Energiae = 0.0; // Energía estable
    double Energiai = 0.0; // Energía inestable

    for (int i = 0; i < NN; i++) {
        if (i == NN - 1) {
            Energiae += (1.0 - cos(ye[i])) + (LAMBDA / 2.0) * pow(ye[0] + 2.0 * M_PI - ye[i], 2.0);
            Energiai += (1.0 - cos(yi[i])) + (LAMBDA / 2.0) * pow(yi[0] + 2.0 * M_PI - yi[i], 2.0);
        } else {
            Energiae += (1.0 - cos(ye[i])) + (LAMBDA / 2.0) * pow(ye[i + 1] - ye[i], 2.0);
            Energiai += (1.0 - cos(yi[i])) + (LAMBDA / 2.0) * pow(yi[i + 1] - yi[i], 2.0);
        }
    }
    return Energiai - Energiae;
}

int main(void) {
    double ye[NN], yi[NN];
    double ve[NN], vi[NN];
    double LAMBDA[A];
    double EPN;
    double Energiae; // (no usada, se mantiene)
    double Energiai; // (no usada, se mantiene)

    FILE *output_file = fopen("EPN_LAMBDA.txt", "w");
    if (output_file == NULL) {
        printf("Error al abrir el archivo.\n");
        return 1;
    }

    for (int i = 0; i < A; i++) {
        LAMBDA[i] = Lambdamin + i * (Lambdamax - Lambdamin) / A;

        for (int k = 0; k < NN; k++) {
            if (k < NN / 2) {
                yi[k] = 0.0;
                ye[k] = 0.0;
            } else if (k == NN / 2) {
                yi[k] = M_PI;
                ye[k] = 0.0;
            } else {
                yi[k] = 2.0 * M_PI;
                ye[k] = 2.0 * M_PI;
            }
            vi[k] = 0.0;
            ve[k] = 0.0;
        }

        evolve_systemestable(ye, ve, LAMBDA[i]);
        evolve_systeminestable(yi, vi, LAMBDA[i]);

        EPN = Energia(ye, yi, LAMBDA[i]);

        fprintf(output_file, "%0.10f\t%0.10f\n", LAMBDA[i], EPN);
        printf("%0.10f\t%0.10f\n", LAMBDA[i], EPN);
    }

    fclose(output_file);
    return 0;
}
