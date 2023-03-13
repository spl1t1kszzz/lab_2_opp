#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include </opt/homebrew/Cellar/libomp/15.0.7/include/omp.h>


const int n = 1500;
const long double t = 0.00001;
const long double epsilon = 0.00001;

void createMatrix(long double *matrix, int size, long double mainData, long double otherData) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i == j)
                *(matrix + i * n + j) = mainData;
            else
                *(matrix + i * n + j) = otherData;
        }
    }
}

long double vector_square_sum(const long double *vector, int size) {
    long double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += *(vector + i) * *(vector + i);
    }
    return sum;
}


void mult_mat_vec(const long double *matrix, const long double *vector, long double *result, int size) {
    for (int i = 0; i < size; ++i) {
        long double partSum = 0;
        for (int j = 0; j < size; ++j) {
            partSum += *(matrix + i * n + j) * *(vector + j);
        }
        *(result + i) = partSum;
    }
}

void mult_vec_digit(long double *vector, long double digit, int size) {
    for (int i = 0; i < size; ++i) {
        *(vector + i) = *(vector + i) * digit;
    }
}

void sub_vect(const long double *left, const long double *right, long double *result, int size) {
    for (int i = 0; i < size; ++i) {
        *(result + i) = *(left + i) - *(right + i);
    }
}

void iteration(long double *x, long double *b, long double *matrix, long double param, int size, long double *result) {
    long double *ax = malloc(sizeof(long double) * size);
    long double *axB = malloc(sizeof(long double) * size);
    mult_mat_vec((const long double *) matrix, x, ax, size);
    sub_vect(ax, b, axB, size);
    mult_vec_digit(axB, param, size);
    sub_vect(x, axB, result, size);
    free(ax);
    free(axB);
}

bool crit(long double *matrix, long double *x, long double *b, long double param, long double b_sqr_sum, int size) {
    long double *ax = malloc(sizeof(long double) * size);
    long double *axB = malloc(sizeof(long double) * size);
    mult_mat_vec((const long double *) matrix, x, ax, size);
    sub_vect(ax, b, axB, size);
    long double axB_sqr_sum = vector_square_sum(axB,size);
    free(ax);
    free(axB);
    return (axB_sqr_sum / b_sqr_sum) < param * param;
}


void set_vector(const long double *src, long double *dest, int size) {
    for (int i = 0; i < size; ++i) {
        *(dest + i) = *(src + i);
    }
}


int main() {
    double itime, ftime, exec_time;
    long double *matrix = malloc(sizeof(long double) * n * n);
    createMatrix(matrix, n, 2.0, 1.0);
    long double *x = calloc(n, sizeof(long double));
    long double *b = malloc(sizeof(long double) * n);
    for (int i = 0; i < n; ++i) {
        *(b + i) = n + 1;
    }
    long double b_sqr_sum = vector_square_sum(b,n);
    long double *nextX = calloc(n, sizeof(long double));
    set_vector(x, nextX, n);
    itime = omp_get_wtime();
    while (!crit(matrix, nextX, b, epsilon, b_sqr_sum, n)) {
        iteration(x, b, matrix, t, n, nextX);
        set_vector(nextX, x, n);

    }
    ftime = omp_get_wtime();
    exec_time = ftime - itime;
    printf("\n\nTime taken is %f", exec_time);
    free(matrix);
    free(x);
    free(nextX);
    free(b);
    return 0;
}
