%nproc=4
%mem=5760MB
%chk=meoh_310.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4361 0.0104 0.0494
C 0.0195 -0.0039 -0.0022
H 1.5799 0.6093 -0.7130
H -0.3280 0.2879 -0.9932
H -0.4163 -0.9706 0.2502
H -0.4043 0.7397 0.6728

