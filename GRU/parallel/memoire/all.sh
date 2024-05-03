#!/bin/bash
rm -rf resultats/*
./run.sh
cd ../../..
cd LSTM/
cd parallel/
cd memoire/
rm -rf resultats/*
./run.sh
cd ..
cd article/
rm -rf resultats/*
./run.sh
