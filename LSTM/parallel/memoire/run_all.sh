#!/bin/bash
make
rm -rf resultats/*
echo -e "\n Nombre de Threads : "
read th
./app.exe -hiden 80 -epoch 15 -batch 32 -lr 0.01 -thread $th
