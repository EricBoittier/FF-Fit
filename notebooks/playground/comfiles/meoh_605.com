%nproc=4
%mem=5760MB
%chk=meoh_605.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4287 0.0292 -0.0711
C 0.0057 0.0091 0.0208
H 1.7411 0.3362 0.8057
H -0.2475 -1.0395 0.1774
H -0.3866 0.6288 0.8271
H -0.3931 0.2868 -0.9549

