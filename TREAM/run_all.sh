#!/bin/bash

for d in */ ; do
  cd ${d}
  if [ -f first_exp.py ]; then
    python3 first_exp.py &
  fi
  cd ..
done
