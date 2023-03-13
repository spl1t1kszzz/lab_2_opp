#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include </opt/homebrew/Cellar/libomp/15.0.7/include/omp.h>

#define n 1500
#define epsilon 0.00001f
#define t 0.00001f
#define b_norm n * (n + 1) * (n + 1)

double* create_matrix(int size) {
    double *matrix = (double*) malloc(sizeof(double) * size * size);
    int i,j;
    #pragma omp parallel for private(j) collapse(2)
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            matrix[i * n + j] = (i == j) ? 2.0 : 1.0;
        }
    }
    return matrix;
}

double vector_square_sum(const double* vector, int size) {
    double result = 0.0;
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        result += vector[i] * vector[i];
    }
    return result;
}

double* mult_mat_vec(const double* matrix, const double* vector, int size) {
    double* result = (double*) malloc(sizeof(double) * size);
    int i,j;
#pragma omp parallel for
    for (i = 0; i < size; ++i) {
        double partial_sum = 0.0;
        for (j = 0; j < size; ++j) {
            partial_sum += matrix[i * n + j] * vector[j];
        }
        result[i] = partial_sum;
    }
    return result;
}

void mult_vector_on_digit(double* vector, double digit, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        vector[i] *= digit;
    }
}


double* sub_vect(const double* left, const double* right, int size) {
    double* result = (double*) malloc(sizeof(double) * size);
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        result[i] = left[i] - right[i];
    }
    return result;
}


double* simple_iteration(double* x, double* b, double* a, int size) {
    double* ax = mult_mat_vec(a, x, size);
    double* ax_b = sub_vect(ax, b, size);
    mult_vector_on_digit(ax_b, t, size);
    double* x_next = sub_vect(x, ax_b, size);
    free(ax);
    free(ax_b);
    return x_next;
}

bool criteria(double* a, double *x, double* b,int size) {
    double* ax = mult_mat_vec(a, x, size);
    double* ax_b = sub_vect(ax, b, size);
    const double ax_b_norm = vector_square_sum(ax_b, size);
    free(ax);
    free(ax_b);
    return (ax_b_norm / b_norm) < epsilon;
}

void fill_vector(double *vector, int size, double number) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        vector[i] = number;
    }
}

int main() {
    double start_time, end_time, exec_time;
    double* a = create_matrix(n);
    double* x = calloc(n, sizeof(double));
    double* next_x;
    double* b = malloc(sizeof(double) * n);
    fill_vector(b, n, (double) (n + 1));
    start_time = omp_get_wtime();
    while (!criteria(a, x, b, n)) {
        next_x = simple_iteration(x, b, a, n);
        memcpy(x, next_x, sizeof(double) * n);
        free(next_x);
    }
    end_time = omp_get_wtime();
    exec_time = end_time - start_time;
    printf("Time taken is %f\n", exec_time);
    printf("%lf", x[0]);
    free(a);
    free(b);
    free(x);
    return 0;
}