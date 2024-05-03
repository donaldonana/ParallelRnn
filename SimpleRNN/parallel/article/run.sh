#!/bin/bash
i=1
echo -e "\nMaximum Threads : "
read th
while [[ $i -le $th ]] ; do
   echo -e "\n-------------------- Execution with $i Thread(s)--------------------\n"  
   ./app.exe -hiden 80 -epoch 15 -batch 32 -lr 0.01 -thread $i
   if [ $i == 1 ]; then
  	(( i += 1 ))
   else
   	(( i += 2 ))
   fi
done
