%nproc=4
%mem=5760MB
%chk=meoh_522.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4328 0.0870 0.0413
C -0.0212 -0.0214 0.0057
H 1.8336 -0.3387 -0.7452
H -0.2954 -0.6357 0.8633
H -0.2882 1.0294 0.1182
H -0.2799 -0.3955 -0.9849

