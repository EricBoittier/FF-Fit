%nproc=4
%mem=5760MB
%chk=meoh_562.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4389 0.1135 -0.0397
C -0.0032 -0.0201 0.0072
H 1.7267 -0.7049 0.4163
H -0.2688 -0.9173 0.5663
H -0.3776 0.8569 0.5353
H -0.4225 -0.0122 -0.9989

