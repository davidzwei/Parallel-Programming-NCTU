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

    // TODO: MPI init

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
    
    long long global_sum = 0;

    // TODO: use MPI_Gather
    long recv_local_num[world_size];
    MPI_Gather(&local_sum, 1, MPI_LONG, recv_local_num, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        for(int i=0; i<world_size; i++)
            global_sum += recv_local_num[i];

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
