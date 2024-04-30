%nproc=4
%mem=5760MB
%chk=meoh_836.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4204 0.0225 -0.0652
C 0.0298 -0.0021 0.0228
H 1.7288 0.4929 0.7374
H -0.3878 -0.0999 -0.9793
H -0.3523 -0.8538 0.5855
H -0.4306 0.9129 0.3955

