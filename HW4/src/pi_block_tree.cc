#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>


int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int dest = 0;
    int tag = 0;
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);

    // long int local_sum = cal(tosses/world_size, world_rank);
    // long int local_sum = 0;
    unsigned seed = world_rank;
    long long local_sum = 0;
    
    for(int i = 0; i<(tosses/world_size); i++){
        double x = (double)rand_r(&seed)/(double)(RAND_MAX - 1.0);
        double y = (double)rand_r(&seed)/(double)(RAND_MAX - 1.0);   

        if((x*x + y*y) <= 1){
            local_sum++;
        }
    }
    
    long int global_sum = 0;

    // TODO: binary tree redunction
    // recv
    // even
    int step = 1;
    while (step < world_size){
        // even
        if (world_rank % step == 0)
        {
            // receive
            if ((world_rank /step)% 2 == 0)
            {
                long int recv_num = 0;
                int source = world_rank + step;
                MPI_Recv(&recv_num, 1, MPI_LONG, source, tag, MPI_COMM_WORLD, &status);
                local_sum += recv_num;
            }
            // send
            else
            {
                int dest = world_rank - step;
                MPI_Send(&local_sum, 1, MPI_LONG, dest, tag, MPI_COMM_WORLD);
            }
        }
        step *= 2;
    }
        

    if (world_rank == 0)
    {
        // TODO: PI result
        global_sum = local_sum;
        pi_result = (4 * (double)global_sum)/(double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}

