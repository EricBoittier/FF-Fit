%nproc=4
%mem=5760MB
%chk=meoh_566.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4418 0.1044 -0.0471
C -0.0179 -0.0111 0.0105
H 1.7464 -0.6401 0.5127
H -0.1871 -0.9541 0.5304
H -0.3939 0.8438 0.5726
H -0.3781 0.0110 -1.0180

