## About Project ? 

The objective of the is to compare and analyze several parallel implementation strategies for training an LSTM Recurrent Neural Network in C. 

## Strategy implemented

1. **Mutex On**: <br/> 
description ....  
2. **Mutex Off** : <br/> 
description ...
3. **xxx** : <br/> 
description ...
<br/> 

## How to Run ? 

To run the currently simple test, use the following command:
```
    make -B run-app
```

**Note**: It is important to use option `-B` when running `make` to
force all files to be recompiled and be sure that all your changes have
been taken into account. we have currently restricted the number of data **entry to 4000**, for this test. 

### Configuring properties

The file `Makefile.config` defines the main properties like strategy used, number of epoch ... etc. 
You can modify this file to change the properties of your. Do not forget to recompile your code after any modification
made to this file.

Note that a property can also be changed from the command line when
calling `make` (without editing the Makefile or Makefile.config files). Here is an example that selects the **Mutex On** strategy and and using **8 thread**:
```
     make -B MUTEX=0 NUM_THREADS=8 run-app
```

## About data ? 

For this application, we use data come from ...  The data set contain ... messages ... 