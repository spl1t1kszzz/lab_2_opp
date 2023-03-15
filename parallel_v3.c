

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include </opt/homebrew/Cellar/libomp/15.0.7/include/omp.h>

#define n 6000
#define t 0.00001f
#define epsilon 0.00001
#define b_norm n * (n + 1) * (n + 1)

double *create_matrix(double *matrix, int size) {
    int i, j;
    #pragma omp parallel for schedule(runtime) private(j) collapse(2)
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            matrix[i * size + j] = (i == j) ? 2.0 : 1.0;
        }
    }
    return matrix;
}


void vector_square_sum(const double *vector, int size, double *sum) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < size; i++) {
    #pragma omp atomic update
        sum[0] += vector[i] * vector[i];
    }
}


void mult_mat_vec(const double *matrix, const double *vector, int size, double *next_x) {
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < size; ++i) {
        double partial_sum = 0.0;
        for (int j = 0; j < size; ++j) {
            partial_sum += matrix[i * size + j] * vector[j];
        }
        next_x[i] = partial_sum;
    }
}

void mult_vector_on_digit(double *vector, double digit, int size) {
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < size; ++i) {
        vector[i] *= digit;
    }
}


void sub_vect(const double *left, const double *right, int size, double *res) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < size; ++i) {
        res[i] = left[i] - right[i];
    }
}


bool simple_iteration(double *x, double *b, double *a, int size, double *next_x, double *sum) {
    mult_mat_vec(a, x, size, next_x);
    sub_vect(next_x, b, size, next_x);
    vector_square_sum(next_x, size, sum);
    if (sum[0] / b_norm < epsilon * epsilon) {
        return false;
    }
    mult_vector_on_digit(next_x, t, size);
    sub_vect(x, next_x, size, next_x);
    return true;
}

void fill_vector(double *vector, int size, double number) {
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < size; i++) {
        vector[i] = number;
    }
}


int main(int argc, char** argv) {
    double start_time, end_time, exec_time;
    double *a = malloc(sizeof(double) * n * n);
    double *x = malloc(n * sizeof(double));
    double *next_x = calloc(n, sizeof(double));
    double *b = malloc(sizeof(double) * n);
    start_time = omp_get_wtime();
    double sum = 0.0;
    create_matrix(a, n);
    fill_vector(b, n, (double) (n + 1));
    fill_vector(x, n, (double) 0);
    while (simple_iteration(x, b, a, n, next_x, &sum)) {
        sum = 0.0;
        memcpy(x, next_x, sizeof(double) * n);
    }
    end_time = omp_get_wtime();
    exec_time = end_time - start_time;
    printf("Time taken is %f\n", exec_time);
    printf("%.20lf", x[0]);
    free(a);
    free(b);
    free(x);
    return 0;
}

