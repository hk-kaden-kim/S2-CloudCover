#!/bin/bash

# Redirect stdout and stderr to log.txt
exec > >(tee -a log_prithvi_test.txt) 2>&1

echo -e "\n\n"
TIME="`date "+%Y-%m-%d %H:%M:%S"`"
echo $TIME : Start!
echo -e "\n\n"

python ./test.py -e 30 -b 8 -lr 1e-6 -bc prithvi_eo_v2_300 -dc UNetDecoder

echo -e "\n\n"
TIME="`date "+%Y-%m-%d %H:%M:%S"`"
echo $TIME : Done !
echo -e "\n\n"
