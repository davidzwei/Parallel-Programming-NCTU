#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int dest = 0;
    int tag = 0;
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);
    // srand(world_rank*time(NULL));
    unsigned seed = world_rank;
    long long local_sum = 0;

    for(int i = 0; i<(tosses/world_size); i++){
        double x = (double)rand_r(&seed)/(double)(RAND_MAX - 1.0);
        double y = (double)rand_r(&seed)/(double)(RAND_MAX - 1.0);   

        if((x*x + y*y) <= 1){
            local_sum++;
        }
    }
    
    // cal(tosses/world_size, world_rank);
    long long global_sum = 0;

    if (world_rank > 0)
    {
        // TODO: handle workers
        MPI_Send(&local_sum, 1, MPI_LONG, dest, tag, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        global_sum = local_sum;
        // TODO: master
        for(int i=1; i<world_size; i++)
        {
            MPI_Recv(&local_sum, 1, MPI_LONG, i, tag, MPI_COMM_WORLD, &status);
            global_sum += local_sum;
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        double pi_result = (4 * (double)global_sum)/(double)tosses;
        // printf("%lld\n", tosses);
        
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
