#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <cpuid.h>
#include <math.h>

// Registres MSR Intel
#define MSR_RAPL_POWER_UNIT_INTEL    0x606
#define MSR_PKG_ENERGY_STATUS_INTEL  0x611

// Registres MSR AMD
#define MSR_RAPL_POWER_UNIT_AMD      0xC0010299
#define MSR_PKG_ENERGY_STATUS_AMD    0xC001029B

#define MATRIX_SIZE 1500


typedef struct {
    int is_intel;
    int is_amd;
    uint64_t power_unit_reg;
    uint64_t energy_status_reg;
    double energy_unit;
} cpu_config;

// Détection du processeur
cpu_config detect_cpu() ;

// Lecture MSR
int read_msr(int fd, uint64_t reg, uint64_t *data);

// Configuration de l'unité d'énergie
int setup_energy_unit(int msr_fd, cpu_config *config) ;

// Mesure d'énergie
double measure_energy(int msr_fd, cpu_config *config, uint64_t *data) ;

// Allocation de matrice
double** allocate_matrix(int size);

// Initialisation de matrice
void initialize_matrix(double** matrix, int size) ;

// Multiplication de matrices
void multiply_matrices(double** A, double** B, double** C, int size);

// Libération de la mémoire
void free_matrix(double** matrix, int size) ;

void initEnergy(int,  cpu_config *config, uint64_t data);

int main() {
    int msr_fd;
    uint64_t data;
    double energy_start, energy_end;
    cpu_config config = detect_cpu();
    char msr_path[256];


    // Vérification du type de processeur
    if (!config.is_intel && !config.is_amd) {
        fprintf(stderr, "Processeur non supporté (ni Intel ni AMD)\n");
        return 1;
    }
    printf("Processeur détecté : %s\n", config.is_intel ? "Intel" : "AMD");

    
    
    // Ouverture du fichier MSR
    snprintf(msr_path, sizeof(msr_path), "/dev/cpu/0/msr");
    msr_fd = open(msr_path, O_RDONLY);
    if (msr_fd < 0) {
        fprintf(stderr, "Erreur ouverture %s\n"
                "Exécutez en tant que root et vérifiez que le module msr est chargé:\n"
                "sudo modprobe msr\n", msr_path);
        return 1;
    }

    // Configuration de l'unité d'énergie
    if (setup_energy_unit(msr_fd, &config) < 0) {
        fprintf(stderr, "Erreur de configuration de l'unité d'énergie\n");
        close(msr_fd);
        return 1;
    }

    // Allocation des matrices
    double** A = allocate_matrix(MATRIX_SIZE);
    double** B = allocate_matrix(MATRIX_SIZE);
    double** C = allocate_matrix(MATRIX_SIZE);
    if (!A || !B || !C) {
        fprintf(stderr, "Erreur d'allocation mémoire\n");
       // goto cleanup;
    }



    // Initialisation des matrices
    printf("Initialisation des matrices %dx%d...\n", MATRIX_SIZE, MATRIX_SIZE);
    initialize_matrix(A, MATRIX_SIZE);
    initialize_matrix(B, MATRIX_SIZE);

    // Mesure de l'énergie initiale
    printf("Début de la multiplication...\n");

    energy_start = measure_energy(msr_fd, &config, &data);


    if (energy_start < 0) {
        fprintf(stderr, "Erreur de mesure d'énergie initiale\n");
      //  goto cleanup;
    }
    // Multiplication des matrices
    multiply_matrices(A, B, C, MATRIX_SIZE);
    // Mesure de l'énergie finale
    
    
    
    
    energy_end = measure_energy(msr_fd, &config, &data);



    if (energy_end < 0) {
        fprintf(stderr, "Erreur de mesure d'énergie finale\n");
       // goto cleanup;
    }

    // Calcul et affichage de l'énergie consommée
    printf("Multiplication terminée\n");
    printf("Énergie consommée: %.6f Joules\n", energy_end - energy_start);

//cleanup:
    // Nettoyage
    free_matrix(A, MATRIX_SIZE);
    free_matrix(B, MATRIX_SIZE);
    free_matrix(C, MATRIX_SIZE);
    close(msr_fd);

    return 0;
}
