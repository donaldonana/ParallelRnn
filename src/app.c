#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"
#include "lstm.h"
#include "std_conf.h"
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>


typedef struct timezone timezone_t;
typedef struct timeval timeval_t;


lstm_rnn *lstm;

Data *data;

static struct timeval _t1, _t2;
static struct timezone _tz;

static unsigned long _temps_residuel = 0;
#define top1() gettimeofday(&_t1, &_tz)
#define top2() gettimeofday(&_t2, &_tz)

pthread_mutex_t mutexRnn;

typedef struct thread_param thread_param;
struct thread_param{  
  lstm_rnn* lstm;
  int start;
  int end;
  float loss;
  float accuracy;
};

void init_cpu_time(void)
{
   top1(); top2();
   _temps_residuel = 1000000L * _t2.tv_sec + _t2.tv_usec -
                     (1000000L * _t1.tv_sec + _t1.tv_usec );
}

unsigned long cpu_time(void) /* retourne des microsecondes */
{
   return 1000000L * _t2.tv_sec + _t2.tv_usec -
           (1000000L * _t1.tv_sec + _t1.tv_usec ) - _temps_residuel;
}
 
void (*strategy)(thread_param*, int);

void mutexOn(thread_param *params,  int treated)
{
   
  pthread_mutex_lock (&mutexRnn);
    gradients_decend(lstm, params->lstm, LR, treated);
  pthread_mutex_unlock (&mutexRnn);

  copy_lstm(lstm, params->lstm);
}

void mutexOff(thread_param *params,  int treated)
{
  
  gradients_decend(lstm, params->lstm, LR, treated);
  copy_lstm(lstm, params->lstm);

}


void *ThreadTrain (void *threadparams)  
{ 
   
  int treated = 0;

  for (int i = params->start; i < params->end; i++)
  {
    lstm_forward(params->lstm, data->X[i], params->lstm->cache, data); // Forward

    params->loss = params->loss + binary_loss_entropy(data->Y[i], params->lstm->probs, data->ycol);
    params->accuracy = accuracy(params->accuracy, data->Y[i],  params->lstm->probs, data->ycol);

    lstm_backforward(params->lstm, data->Y[i], (data->xcol-1), params->lstm->cache);  // Backforward

    sum_gradients(params->lstm);
    treated += 1; 
    
    if(treated==MINI_BATCH_SIZE || i == (params->end -1)) 
    {	
      strategy(params, treated);
      treated = 0;
    }
    lstm_reset_gradient(params->lstm, 0);
  }
  
  pthread_exit (NULL);
}
 

int main(int argc, char **argv)
{
    pthread_mutex_init(&mutexRnn, NULL);
    void *status;

    data = malloc(sizeof(Data));
    load(data); 

    thread_param *threads_params = malloc(sizeof(thread_param)*NUM_THREADS);
    pthread_t    *threads        = malloc(sizeof(pthread_t)*NUM_THREADS);

    pthread_attr_t attr ;
    pthread_attr_init(&attr);  
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    lstm = e_calloc(1, sizeof(lstm_rnn));
    lstm_init(data->ecol, HIDEN_SIZE, data->ycol, lstm); 

    int r, start, end, iter = 0;
    float loss, accuracy;

     
    data->xraw = 4000;  // restreindre les données à 4000 entrées.

    top1();
    while (iter < EPOCH)
    {
      start = 0 ; 
      end = data->xraw/NUM_THREADS ;
      loss = accuracy = 0.0 ;

      printf("\nStart of epoch %d/%d \n", (iter+1) , EPOCH);

      /* Create and start threads */
      for (int i=0; i < NUM_THREADS ; i ++) 
      {

        threads_params[i].lstm  = e_calloc(1, sizeof(lstm_rnn));
        lstm_init(data->ecol, HIDEN_SIZE, data->ycol , threads_params[i].lstm);
        copy_lstm(lstm, threads_params[i].lstm);

        threads_params[i].accuracy = 0.0;
        threads_params[i].loss     = 0.0;
        threads_params[i].start    = start;
        threads_params[i].end      = end;
         
        if(MUTEX)
          strategy = mutexOn;
        else
          strategy = mutexOff;

        r = pthread_create (&threads[i] ,&attr ,ThreadTrain ,(void*)&threads_params[i]) ;
        if (r) 
        {
            printf("ERROR; pthread_create() return code : %d\n", r);
            exit(-1);
        }
         
        start = end ;
        end = end + data->xraw/NUM_THREADS;
        if (i == (NUM_THREADS-1) )
        {
          end = end + data->xraw%NUM_THREADS ;
        }   

      }
      // Free attribute and wait the other threads
      pthread_attr_destroy(&attr);

      for(int t=0; t<NUM_THREADS; t++) 
      {
        r = pthread_join(threads[t], &status);
        if (r) {
        printf("ERROR; return code from pthread_join() is %d\n", r);
          exit(-1);
        }
        loss = loss + threads_params[t].loss ;
        accuracy  = accuracy + threads_params[t].accuracy ;
      }

      printf("--> Train loss : %f || Train Accuracy : %f \n" , loss/(data->xraw), accuracy/(data->xraw)); 
       
      iter += 1 ; 

    }

    top2();
	  unsigned long temps = cpu_time();
    printf("\ntraining time = %ld.%03ldms\n", temps/1000, temps%1000);


}

