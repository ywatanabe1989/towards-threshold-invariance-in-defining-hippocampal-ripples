#!/bin/sh

echo \
'../data/02/day4/movie/02_day4.wmv' \
'../data/02/day1/movie/02_day1.wmv' \
'../data/02/day3/movie/02_day3.wmv' \
'../data/03/day2/movie/03_day2.wmv' \
'../data/03/day3/movie/03_day3.wmv' \
'../data/03/day4/movie/03_day4.wmv' \
'../data/03/day1/movie/03_day1.wmv' \
'../data/04/day2/movie/04_day2.wmv' \
'../data/04/day3/movie/04_day3.wmv' \
'../data/04/day4/movie/04_day4.wmv' \
'../data/04/day1/movie/04_day1.wmv' \
'../data/05/day2/movie/05_day2.wmv' \
'../data/05/day3/movie/05_day3.wmv' \
'../data/05/day4/movie/05_day4.wmv' \
'../data/05/day1/movie/05_day1.wmv' \
'../data/01/day1/movie/01_day1.wmv' \
| xargs -P 18 -n 1 python 03_Mice_Position/trace_a_mouse.py -v
