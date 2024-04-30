%nproc=4
%mem=5760MB
%chk=meoh_734.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4291 0.0818 0.0455
C 0.0173 -0.0043 0.0028
H 1.7098 -0.3376 -0.7947
H -0.2813 -0.8539 -0.6114
H -0.4123 -0.1821 0.9886
H -0.4466 0.9118 -0.3628

