#define _GNU_SOURCE
#include "energy_intel_amd.h"


// Détection du processeur
cpu_config detect_cpu() {
    cpu_config config = {0};
    unsigned int eax, ebx, ecx, edx;
    char vendor[13];

    __cpuid(0, eax, ebx, ecx, edx);
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';

    config.is_intel = strcmp(vendor, "GenuineIntel") == 0;
    config.is_amd = strcmp(vendor, "AuthenticAMD") == 0;

    if (config.is_intel) {
        config.power_unit_reg = MSR_RAPL_POWER_UNIT_INTEL;
        config.energy_status_reg = MSR_PKG_ENERGY_STATUS_INTEL;
    } else if (config.is_amd) {
        config.power_unit_reg = MSR_RAPL_POWER_UNIT_AMD;
        config.energy_status_reg = MSR_PKG_ENERGY_STATUS_AMD;
    }

    return config;
}

// Lecture MSR
int read_msr(int fd, uint64_t reg, uint64_t *data) {
    if (pread(fd, data, sizeof(uint64_t), reg) != sizeof(uint64_t)) {
        perror("Erreur lecture MSR");
        return -1;
    }
    return 0;
}

// Configuration de l'unité d'énergie
int setup_energy_unit(int msr_fd, cpu_config *config) {
    uint64_t data;
    
    if (read_msr(msr_fd, config->power_unit_reg, &data) < 0) {
        return -1;
    }

    if (config->is_intel) {
        config->energy_unit = pow(0.5, (double)((data >> 8) & 0x1F));
    } else if (config->is_amd) {
        // AMD utilise une unité fixe de 15.3 microjoules
        config->energy_unit = 15.3e-6;
    }

    return 0;
}

// Mesure d'énergie
double measure_energy(int msr_fd, cpu_config *config, uint64_t *data) {
    if (read_msr(msr_fd, config->energy_status_reg, data) < 0) {
        return -1;
    }
    return (double)*data * config->energy_unit;
}


// Allocation de matrice
double** allocate_matrix(int size) {
    double** matrix = (double**)malloc(size * sizeof(double*));
    if (!matrix) return NULL;
    
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)malloc(size * sizeof(double));
        if (!matrix[i]) {
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

// Initialisation de matrice
void initialize_matrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// Multiplication de matrices
void multiply_matrices(double** A, double** B, double** C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Libération de la mémoire
void free_matrix(double** matrix, int size) {
    if (!matrix) return;
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


void initEnergy(int *msr_fd,  cpu_config *config){

    char msr_path[256];
    // Vérification du type de processeur
    if (!config->is_intel && !config->is_amd) {
        fprintf(stderr, "Processeur non supporté (ni Intel ni AMD)\n");
        return 1;
    }
    printf("Processeur détecté : %s\n", config->is_intel ? "Intel" : "AMD");
    
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


}

 