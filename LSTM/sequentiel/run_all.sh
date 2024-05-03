#!/bin/bash
make
rm -rf resultats/*
./app.exe -hiden 80 -epoch 15 -batch 32 -lr 0.01 -thread 2
