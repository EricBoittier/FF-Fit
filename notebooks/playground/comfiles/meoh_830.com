%nproc=4
%mem=5760MB
%chk=meoh_830.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4428 0.0342 -0.0680
C -0.0136 0.0008 0.0070
H 1.6910 0.3164 0.8371
H -0.4003 -0.2123 -0.9895
H -0.2360 -0.8219 0.6866
H -0.3348 0.9499 0.4363

