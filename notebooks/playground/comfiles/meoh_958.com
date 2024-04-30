%nproc=4
%mem=5760MB
%chk=meoh_958.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4606 -0.0069 -0.0426
C -0.0196 0.0078 0.0071
H 1.5201 0.8438 0.4405
H -0.3811 1.0246 0.1607
H -0.3190 -0.4909 -0.9146
H -0.3120 -0.5770 0.8792

