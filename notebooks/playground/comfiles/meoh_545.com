%nproc=4
%mem=5760MB
%chk=meoh_545.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4313 0.1107 -0.0029
C -0.0158 -0.0067 0.0123
H 1.8012 -0.7888 -0.1237
H -0.2004 -0.8546 0.6719
H -0.4202 0.9433 0.3617
H -0.2520 -0.1919 -1.0356

