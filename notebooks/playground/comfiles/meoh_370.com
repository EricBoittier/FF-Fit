%nproc=4
%mem=5760MB
%chk=meoh_370.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4347 0.0352 -0.0617
C -0.0266 0.0099 0.0073
H 1.7981 0.1923 0.8348
H -0.3902 0.9893 -0.3037
H -0.2174 -0.7901 -0.7080
H -0.1875 -0.2835 1.0447

