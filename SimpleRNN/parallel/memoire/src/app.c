#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"
#include "simplernn.h"
#include "std_conf.h"
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
// # define NUM_THREADS 2

struct timeval start_t , end_t ;
pthread_mutex_t mutexRnn;
SimpleRnn *rnn;
Data *data ;
float lr , VALIDATION_SIZE ;
int epoch, MINI_BATCH_SIZE , HIDEN_SIZE, EXECUTION = 1, NUM_THREADS = 1;

typedef struct thread_param thread_param;
struct thread_param{  
  SimpleRnn* rnn;
  SimpleRnn* gradient  ;
  SimpleRnn* AVGgradient ;
  int start;
  int end;
  float loss;
  float acc;
};


void parse_input_args(int argc, char** argv)
{
  int a = 0;

  while ( a < argc ) {

    if ( argc <= (a+1) ){ 
      break; // All flags have values attributed to them
    }

    if ( !strcmp(argv[a], "-lr") ) {
      lr = atof(argv[a + 1]);
      if ( lr == 0.0 ) {
        // usage(argv);
      }
    } else if ( !strcmp(argv[a], "-thread") ) {
      NUM_THREADS = atoi(argv[a + 1]);
      if ( NUM_THREADS <= 0 ) {
        // usage(argv);
      }
    } else if ( !strcmp(argv[a], "-epoch") ) {
      epoch = (unsigned long) atoi(argv[a + 1]);
      if ( epoch == 0 ) {
        // usage(argv);
      }
    } else if ( !strcmp(argv[a], "-batch") ) {
      MINI_BATCH_SIZE = (unsigned long) atoi(argv[a + 1]);
      if ( MINI_BATCH_SIZE == 0 ) {
        // usage(argv);
      }
    } else if ( !strcmp(argv[a], "-validation") ) {
      VALIDATION_SIZE =  atof(argv[a + 1]);
      if ( VALIDATION_SIZE < 0.1 || VALIDATION_SIZE > 0.3) {
        // usage(argv);
        VALIDATION_SIZE = 0.1;
      }
      
    } else if ( !strcmp(argv[a], "-hiden") ) {
      HIDEN_SIZE =  atoi(argv[a + 1]);
      if ( HIDEN_SIZE < 4 || HIDEN_SIZE > 500) {
        // usage(argv);
        HIDEN_SIZE = 16;
      }
      
    }else if ( !strcmp(argv[a], "-execution") ) {
      EXECUTION =  atoi(argv[a + 1]);
      if ( EXECUTION < 1) {
        // usage(argv);
        EXECUTION = 1;
      }
    }
    a += 1;
    
  }
}


void *ThreadTrain (void *params) // Thread Code  
{ 
  struct thread_param *mes_param ;
  mes_param = ( struct thread_param *) params ;
  mes_param->AVGgradient = e_calloc(1, sizeof(SimpleRnn));
  rnn_init_model(rnn->X, rnn->N, rnn->Y , mes_param->AVGgradient , 1);
  int nb_traite = 0;
  
  for (int i = mes_param->start ; i < mes_param->end; i++)
  {
    // Forward
    rnn_forward(mes_param->rnn, data->X[i], mes_param->rnn->cache, data);
    // Compute Loss
    mes_param->loss = mes_param->loss + loss_entropy(data->Y[i], mes_param->rnn->probs, data->ycol);
    // Compute Accuracy training
    mes_param->acc = accuracy(mes_param->acc , data->Y[i],  mes_param->rnn->probs, data->ycol);
    // Backforward
    rnn_backforward(mes_param->rnn, data->Y[i], (data->xcol-1), mes_param->rnn->cache, mes_param->gradient);
    sum_gradients(mes_param->AVGgradient, mes_param->gradient);
    nb_traite = nb_traite + 1; 
    // Update The Central RNN
    if(nb_traite==MINI_BATCH_SIZE || i == (mes_param->end - 1) )
    {	
      pthread_mutex_lock(&mutexRnn);
        gradients_decend(rnn, mes_param->AVGgradient, lr, nb_traite);
        copy_rnn(rnn, mes_param->rnn);
      pthread_mutex_unlock(&mutexRnn);
      nb_traite = 0 ;
    }
    rnn_zero_the_model(mes_param->gradient);
  }
  rnn_free_model(mes_param->gradient);
  rnn_free_model(mes_param->AVGgradient);
  pthread_exit (NULL);
}


int main(int argc, char **argv)
{
    srand(time(NULL));
    pthread_mutex_init(&mutexRnn, NULL);
    FILE *fl  = fopen(LOSS_FILE_NAME, "a");
    FILE *fa  = fopen(ACC_FILE_NAME,  "a");
    FILE *ft  = fopen(TEST_FILE_NAME,  "a");
    FILE *ftime  = fopen(TIME_FILE_NAME,  "a"); 

    data = malloc(sizeof(Data));
    double totaltime;
    void *status;
    int n , r, end, start = 0 , size , stop = 0, e = 0 , X , N , Y;
    float Loss , Acc , val_loss, best_loss = 100 ;
    
    // Set Parameters And Retreive data
    lr = 0.01; epoch = 20; NUM_THREADS = 2; MINI_BATCH_SIZE = 16; VALIDATION_SIZE = 0; HIDEN_SIZE = 64; 
    parse_input_args(argc, argv);
    get_split_data(data, VALIDATION_SIZE);
    size = (data->start_val - 1) ; X = data->ecol ; Y = data->ycol; N = HIDEN_SIZE; n = size/NUM_THREADS; 
    SimpleRnn *rnn_copy;
    rnn_copy = e_calloc(1, sizeof(SimpleRnn));
    rnn_init_model(X, N, Y , rnn_copy, 0); 
    
    /* Initialize and Set thread params */
    thread_param *threads_params = malloc(sizeof(thread_param)*NUM_THREADS);
    pthread_t *threads = malloc(sizeof(pthread_t)*NUM_THREADS);
    pthread_attr_t attr ;

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    /* Initialize central Model */
    rnn = e_calloc(1, sizeof(SimpleRnn));
    rnn_init_model(X, N, Y , rnn, 0); 
    print_summary(rnn, epoch, MINI_BATCH_SIZE, lr, NUM_THREADS);

      printf("\n====== Training =======\n");

    gettimeofday(&start_t, NULL);
    while (e < epoch)
    {
      start = 0 ; 
      end = n ;
      Loss = Acc = 0.0 ;
      printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
      /* Create And Start The Threads */
      for ( int i=0; i < NUM_THREADS ; i ++) 
      {
        threads_params[i].rnn = e_calloc(1, sizeof(SimpleRnn));
        rnn_init_model(X, N, Y , threads_params[i].rnn , 0);
        copy_rnn(rnn, threads_params[i].rnn);
        threads_params[i].gradient = e_calloc(1, sizeof(SimpleRnn));
        rnn_init_model(X, N, Y , threads_params[i].gradient , 1);
        threads_params[i].loss = 0.0;
        threads_params[i].acc = 0.0;
        threads_params[i].start = start;
        threads_params[i].end = end;
        r = pthread_create (&threads[i] ,&attr ,ThreadTrain ,(void*)&threads_params[i]) ;
        if (r)
        {
          printf("ERROR; pthread_create() return code : %d\n", r);
          exit(-1);
        }
        start = end ;
        end   = end + n;
        if (i == (NUM_THREADS-1) )
        {
          end = end + size%NUM_THREADS ;
        }
      }
      /* Free attribute and wait for the other threads */
      pthread_attr_destroy(&attr);
      for(int t=0; t<NUM_THREADS; t++)
      {
        r = pthread_join(threads[t], &status);
        if (r)
        {
          printf("ERROR; return code from pthread_join() is %d\n", r);
          exit(-1);
        }
        Loss = Loss + threads_params[t].loss ;
        Acc = Acc + threads_params[t].acc ;
      }
      printf("--> Train Loss : %f || Train Accuracy : %f \n" , Loss/size, Acc/size); 
      fprintf(fl,"%d,%d,%.6f\n", NUM_THREADS, e+1 , Loss/size);
      fprintf(fa,"%d,%d,%.6f\n", NUM_THREADS, e+1, Acc/size);
      /* Validation Phase And Early Stoping */
      val_loss = rnn_validation(rnn, data);
      // fprintf(fv,"%d,%d,%d,%.6f\n", EXECUTION, NUM_THREADS, e+1 , val_loss);
      if (val_loss < best_loss)
      {
        printf("\nsave");
        copy_rnn(rnn, rnn_copy); 
        stop = 0;
        best_loss = val_loss;
      }
      else
      {
        stop = stop + 1;
      }
      e = e + 1 ; 
    }
    gettimeofday(&end_t, NULL);
    totaltime = (((end_t.tv_usec - start_t.tv_usec) / 1.0e6 + end_t.tv_sec - start_t.tv_sec) * 1000) / 1000;
    printf("\nTRAINING PHASE END IN %lf s\n", totaltime);
    fprintf(ftime,"%d,%.2f\n", NUM_THREADS , totaltime);

    pthread_mutex_destroy(&mutexRnn);
    
    printf("\n====== Test Phase ======\n");
    printf(" \n...\n");
    rnn_test(rnn_copy, data, EXECUTION, NUM_THREADS, ft);
    rnn_store_net_layers_as_json(rnn_copy, MODEL_FILE_NAME); 
    printf("\n");

    rnn_free_model(rnn);
    free(threads);
    free(threads_params);
    pthread_exit(NULL);

}

