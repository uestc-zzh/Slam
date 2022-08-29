#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#define N 2

void *thread(void *vargp);

char **ptr;

int main(int argc, char **argv)
{
    int i;
    pthread_t tid;
    char *msgs[N] = {
        "Hello from thread1",
        "Hello from thread2"};

    ptr = msgs;
    for (i = 0; i < N; i++)
    {
        pthread_create(&tid, NULL, thread, (void*)i);
        //sleep(1);
    }
    pthread_exit(NULL);

    return 0;
}

void *thread(void *vargp)
{
    //int myid = *((int*)vargp);
    int myid = (int)vargp;
    static int cnt = 0;
    printf("[%d]:%s(cnt=%d)\n", myid, ptr[myid], ++cnt);
    //printf("[%ld]\n",*((long*)myid));
    return NULL;
}