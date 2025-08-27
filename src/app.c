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

timeval_t t1, t2;
timezone_t tz;

lstm_rnn *lstm;

Data *data;

static struct timeval _t1, _t2;
static struct timezone _tz;
timeval_t t1, t2;
timezone_t tz;

static unsigned long _temps_residuel = 0;
#define top1() gettimeofday(&_t1, &_tz)
#define top2() gettimeofday(&_t2, &_tz)

pthread_mutex_t mutexRnn;

typedef struct thread_param thread_param;
struct thread_param{  
  lstm_rnn* lstm;
  lstm_rnn* gradient;
  lstm_rnn* AVGgradient;
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

void initThread(thread_param *params, int end, int start)
{
  params->lstm     = e_calloc(1, sizeof(lstm_rnn));
  params->gradient = e_calloc(1, sizeof(lstm_rnn));

  lstm_init(data->ecol, HIDEN_SIZE, data->ycol , params->lstm, 0);
  lstm_init(data->ecol, HIDEN_SIZE, data->ycol,  params->gradient, 1);
  copy_lstm(lstm, params->lstm);

  params->accuracy = 0.0;
  params->loss     = 0.0;
  params->start    = start;
  params->end      = end;

}

void *ThreadTrain (void *threadparams)  
{ 
  struct thread_param *params ;
  params = (struct thread_param *) threadparams ;
  params->AVGgradient = e_calloc(1, sizeof(lstm_rnn));

  lstm_init(lstm->X, lstm->N, lstm->Y , params->AVGgradient , 1);
  
  int nb_traite = 0;

  for (int i = params->start; i < params->end; i++)
  {
     // Forward
    lstm_forward(params->lstm, data->X[i], params->lstm->cache, data);
    
    // Compute loss
    params->loss = params->loss + binary_loss_entropy(data->Y[i], params->lstm->probs, data->ycol);
    
    //  Accuracy
    params->accuracy = accuracy(params->accuracy , data->Y[i],  params->lstm->probs, data->ycol);
    
    // Backforward
    lstm_backforward(params->lstm, data->Y[i], (data->xcol-1), params->lstm->cache, params->gradient);
    
    sum_gradients(params->AVGgradient, params->gradient);
    
    nb_traite += 1; 

    // Update The Central LSTM
    if(nb_traite==MINI_BATCH_SIZE || i == (params->end -1))
    {	
      pthread_mutex_lock (&mutexRnn);
        gradients_decend(lstm, params->AVGgradient, LR, nb_traite);
        copy_lstm(lstm, params->lstm);
      pthread_mutex_unlock (&mutexRnn);
      nb_traite = 0;
    }

    lstm_zero_the_model(params->gradient);
    set_vector_zero(lstm->h_prev, lstm->N);
    set_vector_zero(lstm->c_prev, lstm->N);
    
  }
  lstm_free_model(params->gradient);
  lstm_free_model(params->AVGgradient);
  
  pthread_exit (NULL);

}

void *ThreadTrainMutexOff (void *threadparams) // Code du thread
{ 
  struct thread_param *params ;
  params = (struct thread_param *) threadparams ;
  params->AVGgradient = e_calloc(1, sizeof(lstm_rnn));
  
  lstm_init(lstm->X, lstm->N, lstm->Y , params->AVGgradient , 1);
  int nb_traite = 0;

  for (int i = params->start; i < params->end; i++)
  {
    // Forward
    lstm_forward(params->lstm, data->X[i], params->lstm->cache, data);
    
    // Compute loss
    params->loss = params->loss + binary_loss_entropy(data->Y[i], params->lstm->probs, data->ycol);
    
    //  Accuracy
    params->accuracy = accuracy(params->accuracy , data->Y[i],  params->lstm->probs, data->ycol);
    
    // Backforward
    lstm_backforward(params->lstm, data->Y[i], (data->xcol-1), params->lstm->cache, params->gradient);
    
    sum_gradients(params->AVGgradient, params->gradient);
    
    nb_traite += 1; 
    
    // Update The Central LSTM
    if(nb_traite==MINI_BATCH_SIZE || i == (params->end -1))
    {	
      //pthread_mutex_lock (&mutexRnn);
        gradients_decend(lstm, params->AVGgradient, LR, nb_traite);
        copy_lstm(lstm, params->lstm);
      //pthread_mutex_unlock (&mutexRnn);
      nb_traite = 0;
    }
    lstm_zero_the_model(params->gradient);
    set_vector_zero(lstm->h_prev, lstm->N);
    set_vector_zero(lstm->c_prev, lstm->N);
    
  }
  lstm_free_model(params->gradient);
  lstm_free_model(params->AVGgradient);
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
    lstm_init(data->ecol, HIDEN_SIZE, data->ycol, lstm, 0); 

    int r, start, end, iter = 0;
    float loss, accuracy;

    // restreindre les données à 4000 entrées. 
    data->xraw = 4000; 

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
        initThread(&threads_params[i], end, start);
        
        if(MUTEX)
        	r = pthread_create (&threads[i] ,&attr ,ThreadTrain ,(void*)&threads_params[i]) ;
        else
          r = pthread_create (&threads[i] ,&attr ,ThreadTrainMutexOff ,(void*)&threads_params[i]) ;
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

