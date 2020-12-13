#include <stdio.h>
#include <iostream>
#include <mpi.h>
#include <string>
#include <chrono>
#include <time.h>

#define MASTER 0


using namespace std;


double rand_double(double fMin, double fMax)
{
    double f = (double) rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


double rand_int(int fMin, int fMax)
{
    return (double) ( fMin + (std::rand() % (fMax - fMin + 1)) );
}


double** alloc_2d(int rows, int cols) {
    double *data = (double *) malloc(rows * cols * sizeof(double));
    double **array= (double **) malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        array[i] = &(data[cols*i]);
    return array;
}


double** createRandomMatrix(unsigned width, unsigned height)
{
    double** matrix = alloc_2d(width, height);

    for (int w = 0; w < width; w++)
    {
        for (int h = 0; h < height; h++)
        {
            matrix[w][h] = rand_int(0, 10);
        }
    }
    return matrix;
}


void print_matrix(double** A, int A_rows, int A_columns, std::string name) 
{
    std::cout << name << std::endl;

    for(int i = 0; i < A_rows; i++)
    {
        for(int j = 0; j < A_columns; j++)
        {
            std::cout << " " << A[i][j];
        }
        std::cout << std::endl;
    }
}


double** serial_matrix_multiplication(double** A, double** B, int dim) 
{
    double** C = new double*[dim];

    for (int i = 0; i < dim; i++) {
        C[i] = new double[dim];

        for (int j = 0; j < dim; j++) {
            C[i][j] = (double) 0;

            for (int k = 0; k < dim; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}


void run_serial() 
{
    double** A = createRandomMatrix(2, 3);
    double** B = createRandomMatrix(3, 2);
    double** C = serial_matrix_multiplication(A, B, 2);

    print_matrix(A, 2, 3, "A");
    print_matrix(B, 3, 2, "B");
    print_matrix(C, 2, 2, "C");
}


void send_matrix(double** mat, int rows, int columns, int proc)
{
    for (int row = 0; row < rows; row++) 
        MPI_Send(&mat[row][0], columns, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);   
}

double** receive_matrix(int rows, int columns, int proc)
{
    double** mat = alloc_2d(rows, columns);

    for (int row = 0; row < rows; row++)
        MPI_Recv(&mat[row][0], columns, MPI_DOUBLE, proc, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    return mat;
}

void send_arr(int* arr, int len, int proc)
{
    MPI_Send(&arr[0], len, MPI_INT, proc, 0, MPI_COMM_WORLD);   
}

int* receive_arr(int len, int proc)
{
    int* arr = new int[len];
    MPI_Recv(&arr[0], len, MPI_INT, proc, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return arr;
}


void parallel_matrix_multiplication_1(double** A, double** B, int dim, int argc, char* argv[]) 
{
    MPI_Init(&argc, &argv);
    int proc_rank, proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    if ((proc_num - 1) == 0) {
        return;
    }

    if (proc_rank == MASTER)
    {
        int A_rows = dim / (proc_num - 1);
        int* size_buf = new int[2];
        double* row_buf = new double[dim];

        for (int proc = 1; proc < proc_num; proc++) 
        {
            size_buf[0] = A_rows;
            size_buf[1] = dim;
            MPI_Send(&size_buf[0], 2, MPI_INT, proc, 0, MPI_COMM_WORLD);

            for (int loc_row = 0; loc_row < A_rows; loc_row++) 
            {
                int gl_row = (proc - 1) * A_rows + loc_row;
                for (int column = 0; column < dim; column++) 
                    row_buf[column] = A[gl_row][column];

                MPI_Send(&row_buf[0], dim, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
            }

            for (int row = 0; row < dim; row++)
            {
                for (int column = 0; column < dim; column++)
                    row_buf[column] = B[row][column];

                MPI_Send(&row_buf[0], dim, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
            }
        }

        double** C = alloc_2d(dim, dim);

        for (int proc = 1; proc < proc_num; proc++) 
        {
            MPI_Status Status;

            for (int loc_row = 0; loc_row < A_rows; loc_row++) {
                MPI_Recv(&row_buf[0], dim, MPI_DOUBLE, proc, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);

                int gl_row = (proc - 1) * A_rows + loc_row;
                for (int column = 0; column < dim; column++) 
                    C[gl_row][column] = row_buf[column];
            }
        }

        //print_matrix(C, dim, dim, "Output");
    }
    else 
    {
        MPI_Status Status;
        int* size_buf = new int[2];
        MPI_Recv(&size_buf[0], 2, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);

        int A_rows = size_buf[0];
        int dim = size_buf[1];

        double* row_buf = new double[dim];
        double** A_part = alloc_2d(A_rows, dim);
        double** B = alloc_2d(dim, dim);

        for (int row = 0; row < A_rows; row++) 
        {
            MPI_Recv(&row_buf[0], dim, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
            for (int column = 0; column < dim; column++) 
                A_part[row][column] = row_buf[column];            
        }

        for (int row = 0; row < dim; row++) 
        {
            MPI_Recv(&row_buf[0], dim, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
            for (int column = 0; column < dim; column++) 
                B[row][column] = row_buf[column];            
        }

        double** C_part = alloc_2d(A_rows, dim);

        for (int i = 0; i < A_rows; i++) {
            for (int j = 0; j < dim; j++) {
                C_part[i][j] = (double) 0;

                for (int k = 0; k < dim; k++) 
                    C_part[i][j] += A_part[i][k] * B[k][j];
            }
        }

        for (int row = 0; row < A_rows; row++)
        {
            for (int column = 0; column < dim; column++)
                row_buf[column] = C_part[row][column];

            MPI_Send(&row_buf[0], dim, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
}


void parallel_matrix_multiplication_2(double** A, double** B, int dim, int argc, char* argv[]) 
{
    MPI_Init(&argc, &argv);
    int proc_rank, proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    if ((proc_num - 1) == 0) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return;
    }

    if (proc_rank == MASTER)
    {
        //print_matrix(A, dim, dim, "A");
        //print_matrix(B, dim, dim, "B");

        MPI_Datatype column_type; 
        MPI_Type_vector(dim, 1, dim, MPI_DOUBLE, &column_type);
        MPI_Type_commit(&column_type);

        int B_columns = dim / (proc_num - 1);
        int* size_buf = new int[2] { B_columns, dim };

        for (int proc = 1; proc < proc_num; proc++) 
        {
            send_arr(size_buf, 2, proc);
            send_matrix(A, dim, dim, proc);

            for (int loc_column = 0; loc_column < B_columns; loc_column++)
            {
                int gl_column = (proc - 1) * B_columns + loc_column;
                MPI_Send(&B[0][gl_column], 1, column_type, proc, 0, MPI_COMM_WORLD);
            }

            //print_matrix(receive_matrix(dim, dim, proc), dim, dim, "A_out");
            //print_matrix(receive_matrix(dim, B_columns, proc), dim, B_columns, "B_part_out");
        }

        double** C = alloc_2d(dim, dim);

        for (int proc = 1; proc < proc_num; proc++) 
        {
            double** C_part = receive_matrix(dim, B_columns, proc);

            for (int loc_column = 0; loc_column < B_columns; loc_column++) {
                int gl_column = (proc - 1) * B_columns + loc_column;

                for (int row = 0; row < dim; row++)
                    C[row][gl_column] = C_part[row][loc_column];
            }

            //print_matrix(C_part, dim, B_columns, "C_part");

            free(C_part);
        }

        //print_matrix(C, dim, dim, "Output");
    }
    else 
    {
        int* size_buf = receive_arr(2, MASTER);
        int B_columns = size_buf[0];
        int dim = size_buf[1];
        free(size_buf);

        double* column_buf = new double[dim];
        double** A = receive_matrix(dim, dim, MASTER);
        double** B_part = alloc_2d(dim, B_columns);

        for (int column = 0; column < B_columns; column++) 
        {
            MPI_Recv(&column_buf[0], dim, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int row = 0; row < dim; row++) 
                B_part[row][column] = column_buf[row];            
        }

        //send_matrix(A, dim, dim, MASTER);
        //send_matrix(B_part, dim, B_columns, MASTER);

        double** C_part = alloc_2d(dim, B_columns);

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < B_columns; j++) {
                C_part[i][j] = (double) 0;

                for (int k = 0; k < dim; k++) 
                    C_part[i][j] += A[i][k] * B_part[k][j];
            }
        }

        send_matrix(C_part, dim, B_columns, MASTER);
    }

    MPI_Finalize();
}


void box(string name, unsigned int time) {
	cout << name << endl;
	cout << "Runtime: " << time << endl;
	cout << endl;
}


void run_serial_benchmark(double** A, double** B, int n) {
	auto start = chrono::high_resolution_clock::now();
	double** result = serial_matrix_multiplication(A, B, n);
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
	box("Serial", duration.count());
}


void run_parallel_1_benchmark(double** A, double** B, int n, int argc, char* argv[]) {
	auto start = chrono::high_resolution_clock::now();
	parallel_matrix_multiplication_1(A, B, n, argc, argv);
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
	box("Parallel 1", duration.count());
}


void run_parallel_2_benchmark(double** A, double** B, int n, int argc, char* argv[]) {
	auto start = chrono::high_resolution_clock::now();
	parallel_matrix_multiplication_2(A, B, n, argc, argv);
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
	box("Parallel 2", duration.count());
}


int main (int argc, char* argv[])
{
    int dim = 1024;
    double** A = createRandomMatrix(dim, dim);
    double** B = createRandomMatrix(dim, dim);

    //run_serial_benchmark(A, B, dim);
    //run_parallel_1_benchmark(A, B, dim, argc, argv);
    run_parallel_2_benchmark(A, B, dim, argc, argv);

    //parallel_matrix_multiplication_2(A, B, dim, argc, argv);
}
