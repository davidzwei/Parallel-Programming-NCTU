#include <stdlib.h>
#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fastPRNG.h"

# define MAX 1.0
# define MIN -1.0
using namespace fastPRNG;

typedef struct
{
   int thread_id;
   int start;
   int end;
   long long *global_sum;
} Arg; // 傳入 thread 的參數型別


pthread_mutex_t mutexsum;    // pthread 互斥鎖

// 每個 thread 要做的任務
void *count_pi(void *arg)
{
    Arg *data = (Arg *)arg;
    int thread_id = data->thread_id;
    int start = data->start;
    int end = data->end;
    //    double *pi = data->pi;

    // 將原本的 PI 算法切成好幾份
    double x;
    // double local_pi = 0;
    long long int local_sum = 0;
    long long int *global_sum = data->global_sum;
    //    double step = 1 / MAGNIFICATION;

    // unsigned int seed = time(NULL); 
     unsigned int seed = thread_id; 
    // unsigned int seed = 1000;
    fastXS64 fastR;
    for(int i =  start; i<end; i++){
        // double x = (double)rand_r(&seed)/(double)(RAND_MAX - 1.0);
        // double y = (double)rand_r(&seed)/(double)(RAND_MAX - 1.0);
        //double x = (double)(MAX - MIN) * rand_r(&seed) / (double)(RAND_MAX + 1.0) + MIN;
        //double y = (double)(MAX - MIN) * rand_r(&seed) / (double)(RAND_MAX + 1.0) + MIN;
	
	double x = fastR.xoshiro256p_VNI<double>();
	double y = fastR.xoshiro256p_VNI<double>();
        if ( x*x + y*y <= 1.0)
            local_sum++;
    }

    // **** 關鍵區域 ****
    // 一次只允許一個 thread 存取
    pthread_mutex_lock(&mutexsum);
    // 將部分的 PI 加進最後的 PI
    *global_sum += local_sum;
    pthread_mutex_unlock(&mutexsum);
    // *****************

     //printf("Thread %d did %d to %d:  local Pi=%lld global Pi=%lld \n", 
             //thread_id, start, end, local_sum, *global_sum);

    //std::cout << thread_id <<", "<< start<< ", "<<end<< ", "<<local_sum<< ", "<<*global_sum << std::endl;

    pthread_exit((void *)0);
}


int main(int argc, char* argv[])
{
    if(argc != 3)
        exit(1);

    int num_thread = atoi(argv[1]);
    long long toss = atoll(argv[2]);

    pthread_t callThd[num_thread]; // 宣告建立 pthread

    // 初始化互斥鎖
    pthread_mutex_init(&mutexsum, NULL);

    // 設定 pthread 性質是要能 join
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    long long int *global_sum = new long long int;
    *global_sum = 0;

    int part = toss / num_thread;

    Arg arg[num_thread]; // 每個 thread 傳入的參數
    for (int i = 0; i < num_thread; i++)
    {
        // 設定傳入參數
        arg[i].thread_id = i;
        arg[i].start = part * i;
        arg[i].end = part * (i + 1);
        arg[i].global_sum = global_sum; // PI 的指標，所有 thread 共用

        // 建立一個 thread，執行 count_pi 任務，傳入 arg[i] 指標參數
        // pthread_create(&callThd[i], &attr, count_pi, (void *)&arg[i]);
        if(pthread_create(&callThd[i], &attr, count_pi, (void *)&arg[i])!=0)
        {
            std::cerr<<"create thread error"<<std::endl;
            return 1;
        }
    }

    // 回收性質設定
    pthread_attr_destroy(&attr);

    void *status;
    for (int i = 0; i < num_thread; i++)
    {
        // 等待每一個 thread 執行完畢
        pthread_join(callThd[i], &status);
    }

    double pi = 4.0 * (double)(*global_sum) / (double)toss;



    // 所有 thread 執行完畢，印出 PI
    // printf("Pi =  %.10lf \n", *pi);
    // std::cout << pi << std::endl;
    printf("%.10lf\n", pi);

    // 回收互斥鎖
    pthread_mutex_destroy(&mutexsum);
    // 離開
    pthread_exit(NULL);

}

// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
// #include <assert.h>
// #include <pthread.h>
// #include <string.h>

// static unsigned int total_thr;
// static unsigned int compute_cnt;

// typedef struct
// {
//     unsigned int threadId;
//     double partial_sum;
// } WorkerArgs;

// void Thread_work(void params) {
//     WorkerArgs arg = (WorkerArgs)params;
//     double x, y, distance_squared;
//     unsigned long long number_in_circle = 0;
//     unsigned int threadId = arg->threadId;

//     for (size_t i = threadId; i < compute_cnt; i+=total_thr)
//     {
//         x = (double)rand_r(&threadId)/RAND_MAX;
//         y = (double)rand_r(&threadId)/RAND_MAX;

//         distance_squared = x * x + y * y;
//         if (distance_squared <= 1) number_in_circle++;
//     }

//     arg->partial_sum = (number_in_circle << 2)/((double)compute_cnt);

//     return NULL;
// }

// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
// #include <assert.h>
// #include <pthread.h>
// #include <string.h>

// static unsigned int total_thr;
// static unsigned int compute_cnt;

// typedef struct
// {
//     unsigned int threadId;
//     double partial_sum;
// } WorkerArgs;

// # define MAX 1.0
// # define MIN -1.0

// void *Thread_work(void* params) {
//     WorkerArgs *arg = (WorkerArgs*)params;
//     double x, y, distance_squared;
//     unsigned long long number_in_circle = 0;
//     unsigned int threadId = arg->threadId;

//     for (size_t i = threadId; i < compute_cnt; i+=total_thr)
//     {
//         // x = (double)rand_r(&threadId)/RAND_MAX;
//         // y = (double)rand_r(&threadId)/RAND_MAX;
//         double x = (double)(MAX - MIN) * rand_r(&threadId) / (double)(RAND_MAX + 1.0) + MIN;
//         double y = (double)(MAX - MIN) * rand_r(&threadId) / (double)(RAND_MAX + 1.0) + MIN;

//         distance_squared = x * x + y * y;
//         if (distance_squared <= 1) number_in_circle++;
//     }

//     arg->partial_sum = (number_in_circle << 2)/((double)compute_cnt);

//     return NULL;
// }

// int main(int argc, char *argv[])
// {
//     assert(argc == 3 && "./pi.out thread_counts monte_counts");

//     total_thr = atoi(argv[1]);
//     compute_cnt = atoi(argv[2]);

//     WorkerArgs mArgs[total_thr];

//     pthread_t* thread_handles;
//     thread_handles = (pthread_t*)malloc(total_thr*sizeof(pthread_t));
//     if(!thread_handles)
//     {
//         printf("malloc failed\n");
//         return -1;
//     }

//     for(int i = 0; i < total_thr; i++)
//     {
//         mArgs[i].threadId = i;
//         pthread_create(&thread_handles[i], NULL, Thread_work, (void *)&mArgs[i]);
//     }

//     for(int i = 0; i < total_thr; i++) {
//         pthread_join(thread_handles[i], NULL);
//     }

//     double pi_estimate = 0;
//     for(int i = 0; i < total_thr; i++) {
//         pi_estimate+=mArgs[i].partial_sum;
//     }
//     printf("%.10f\n", pi_estimate);

//     free(thread_handles);

//     return 0;
// }
