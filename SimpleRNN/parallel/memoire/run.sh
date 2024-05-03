#!/bin/bash
i=28
echo -e "\nMaximum Threads : "
read th
while [[ $i -le $th ]] ; do
   echo -e "\n-------------------- Execution with $i Thread(s)--------------------\n"  
   ./app.exe -hiden 80 -epoch 15 -batch 32 -lr 0.01 -thread $i
   if [ $i == 1 ]; then
  	(( i += 3 ))
   else
   	(( i += 4 ))
   fi
done
