#!/bin/bash
MAXTH=12
 
HIDEN=(20 40 60)

MUTEX=(0 1)

for mutex in "${MUTEX[@]}"; do

	rm -r "$mutex.mutex"

	mkdir -p "$mutex.mutex"
	
	for hiden in "${HIDEN[@]}"; do
	
		THREAD=1

		while [[ $THREAD -le $MAXTH ]] ; do
		
		   echo -e "\n-------------------- Execution with $THREAD Thread(s) with mutex $mutex --------------------\n"  
		   
		   cpu-energy-meter -r >> "$mutex.mutex/$hiden-energy.txt" & 
		   METER_PID=$!
		   
		   echo -e $THREAD >> "$mutex.mutex/$hiden-energy.txt" 
		   ./app.exe -hiden $hiden -epoch 10 -batch 32 -lr 0.01 -thread $THREAD -mutex $mutex
		   kill -SIGINT $METER_PID
		   
		   if [ $THREAD == 1 ]; then
		  	(( THREAD += 1 ))
		   else
		   	(( THREAD += 2 ))
		   fi
		   
		done

	done

done

 
