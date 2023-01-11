#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <immintrin.h>

#define MASTER 0
#define FROM_MASTER 1 
#define FROM_WORKER 2 

void print_matrix(int row, int col, int *matrix);
void cal_matrix(int *a_matrix, int *b_matrix, int *c_matrix, int n, int m, int l);

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0)
    {
        if(scanf("%d %d %d", n_ptr, m_ptr, l_ptr) != 3)
            exit(1);
        int n = *n_ptr, m = *m_ptr, l = *l_ptr;
        *a_mat_ptr = (int *)malloc(n * m * sizeof(int));
        *b_mat_ptr = (int *)malloc(m * l * sizeof(int));
        // read matrix
        // matrix a
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                scanf("%d", *a_mat_ptr + i * m + j);
            }
        }
        // matrix b
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < l; j++)
            {
                scanf("%d", *b_mat_ptr + i * l + j);
            }
        }
    }
}

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // master
    if (world_rank == 0)
    {
        for (int dest = 1; dest < world_size; dest++)
        {
            MPI_Send(&n, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
        }
        int *mat_c = (int *)malloc(sizeof(int) * n * l);

        int num_workers = (n < world_size) ? n : world_size;
        int avg_row = n / num_workers;
        int extra_row = n % num_workers;
        int offset = 0; // start from which row in A, C
        // master
        int rows = (extra_row) ? avg_row + 1 : avg_row;
        offset += rows;
        
        for (int dest = 1; dest < num_workers; dest++)
        {
            int rows_per_worker = (dest < extra_row) ? avg_row + 1 : avg_row;
            MPI_Send(&rows_per_worker, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(a_mat + offset * m, rows_per_worker * m, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(b_mat, m * l, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            offset = offset + rows_per_worker;
        }
        
        // cal_matrix((int *)a_mat, (int *)b_mat, (int *)mat_c, rows, m, l);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < l; j++)
            {
                mat_c[i * l + j] = 0;
                for (int k = 0; k < m; k++)
                {
                    mat_c[i * l + j] += a_mat[i * m + k] * b_mat[l * k + j];
                }
            }
        }

        /* Receive results from worker tasks */
        MPI_Status status;
        
        for (int source = 1; source < num_workers; source++)
        {
            int rows_per_worker;
            MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows_per_worker, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(mat_c + offset * l, rows_per_worker * l, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
        }
        print_matrix(n, l, mat_c);
        free(mat_c);
    }
    // worker
    else if (world_rank > 0)
    {
        int n;
        // int FROM_MASTER = FROM_MASTER;
        MPI_Status status;
        MPI_Recv(&n, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        
        //receive matrix info
        int offset;
        int rows_per_worker;
        int m, l;
        MPI_Recv(&rows_per_worker, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&m, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&l, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        int *matrix_a = (int *)malloc(sizeof(int) * rows_per_worker * m);
        int *matrix_b = (int *)malloc(sizeof(int) * m * l);
        int *matrix_c = (int *)malloc(sizeof(int) * rows_per_worker * l);
        MPI_Recv(matrix_a, rows_per_worker * m, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(matrix_b, m * l, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        
        // cal_matrix(matrix_a, matrix_b, matrix_c, rows_per_worker, m, l);
        for (int i = 0; i < rows_per_worker; i++)
        {
            for (int j = 0; j < l; j++)
            {
                matrix_c[i * l + j] = 0;
                for (int k = 0; k < m; k++)
                {
                    matrix_c[i * l + j] += matrix_a[i * m + k] * matrix_b[l * k + j];
                }
            }
        }
        // send result to master
        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows_per_worker, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(matrix_c, rows_per_worker * l, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        free(matrix_a);
        free(matrix_b);
        free(matrix_c);
    }
}

void cal_matrix(int *a_matrix, int *b_matrix, int *c_matrix, int n, int m, int l)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < l; j++)
        {
            c_matrix[i * l + j] = 0;
            for (int k = 0; k < m; k++)
            {
                c_matrix[i * l + j] += a_matrix[i * m + k] * b_matrix[l * k + j];
            }
        }
    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0)
    {
        free(a_mat);
        a_mat = NULL;
        free(b_mat);
        b_mat = NULL;
    }
}

void print_matrix(int row, int col, int *matrix)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d ", matrix[i * col + j]);
        }
        printf("\n");
    }
}

int main () {
    int n, m, l;
    int *a_mat, *b_mat;

    MPI_Init(NULL, NULL);
    double start_time = MPI_Wtime();

    construct_matrices(&n, &m, &l, &a_mat, &b_mat);
    matrix_multiply(n, m, l, a_mat, b_mat);
    destruct_matrices(a_mat, b_mat);

    double end_time = MPI_Wtime();
    MPI_Finalize();
    printf("MPI running time: %lf Seconds\n", end_time - start_time);

    return 0;
}
