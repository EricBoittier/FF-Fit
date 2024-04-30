%nproc=4
%mem=5760MB
%chk=meoh_636.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4317 -0.0078 -0.0382
C 0.0067 0.0073 0.0174
H 1.6957 0.8630 0.3257
H -0.2792 -1.0438 -0.0214
H -0.3906 0.4631 0.9243
H -0.3712 0.5396 -0.8555

