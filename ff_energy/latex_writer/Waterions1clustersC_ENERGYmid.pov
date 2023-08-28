#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -9.33*x up 9.00*y
  direction 50.00*z
  location <0,0,50.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7}
#declare pale = finish {ambient 0.5 diffuse 0.85 roughness 0.001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 specular 0.5 }
#declare jmol = finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.05 diffuse 0.3 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.01 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.050;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

// no cell vertices
atom(< -3.75,   1.11,  -3.83>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -3.76,   1.72,  -3.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -2.84,   0.79,  -3.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -0.81,   1.96,  -1.01>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -0.28,   1.54,  -1.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -0.21,   2.26,  -0.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  3.87,   1.31,  -5.45>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  3.05,   1.37,  -4.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  4.32,   0.70,  -4.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  1.14,  -2.40,  -4.58>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  0.84,  -2.88,  -5.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  0.41,  -1.77,  -4.34>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  3.08,  -0.80,  -3.21>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  2.36,  -1.33,  -3.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  2.56,  -0.23,  -2.64>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -3.91,  -2.60,  -1.45>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -2.96,  -2.53,  -1.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -4.32,  -2.47,  -2.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(< -1.69,   2.50,  -5.84>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(< -2.43,   2.16,  -6.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -1.40,   1.75,  -5.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -1.81,  -3.97,  -4.82>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -1.18,  -3.91,  -4.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -1.34,  -4.16,  -5.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -0.93,  -1.31,  -7.04>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(< -1.87,  -1.60,  -6.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(< -0.71,  -0.82,  -6.18>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  1.16,  -0.59,  -0.92>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  0.35,  -0.98,  -1.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  0.91,  -0.48,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -1.05,  -2.17,  -1.75>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -1.05,  -1.54,  -2.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -0.73,  -3.03,  -2.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  1.63,   1.99,  -3.75>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  1.25,   2.73,  -4.21>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  0.97,   1.28,  -3.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(< -3.59,   2.19,  -1.43>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(< -3.87,   2.42,  -0.50>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(< -2.66,   2.43,  -1.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -0.70,  -0.12,  -4.03>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #39
cylinder {< -3.75,   1.11,  -3.83>, < -3.30,   0.95,  -3.84>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.84,   0.79,  -3.85>, < -3.30,   0.95,  -3.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.75,   1.11,  -3.83>, < -3.76,   1.41,  -3.43>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.76,   1.72,  -3.03>, < -3.76,   1.41,  -3.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.81,   1.96,  -1.01>, < -0.51,   2.11,  -0.66>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.21,   2.26,  -0.31>, < -0.51,   2.11,  -0.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.81,   1.96,  -1.01>, < -0.54,   1.75,  -1.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.28,   1.54,  -1.69>, < -0.54,   1.75,  -1.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.87,   1.31,  -5.45>, <  3.46,   1.34,  -5.21>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.05,   1.37,  -4.97>, <  3.46,   1.34,  -5.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.87,   1.31,  -5.45>, <  4.09,   1.00,  -5.17>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.32,   0.70,  -4.88>, <  4.09,   1.00,  -5.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.14,  -2.40,  -4.58>, <  0.99,  -2.64,  -4.99>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.84,  -2.88,  -5.41>, <  0.99,  -2.64,  -4.99>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.14,  -2.40,  -4.58>, <  0.78,  -2.08,  -4.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.41,  -1.77,  -4.34>, <  0.78,  -2.08,  -4.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.41,  -1.77,  -4.34>, < -0.14,  -0.95,  -4.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.70,  -0.12,  -4.03>, < -0.14,  -0.95,  -4.18>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.08,  -0.80,  -3.21>, <  2.72,  -1.06,  -3.40>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.36,  -1.33,  -3.58>, <  2.72,  -1.06,  -3.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.08,  -0.80,  -3.21>, <  2.82,  -0.51,  -2.92>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.56,  -0.23,  -2.64>, <  2.82,  -0.51,  -2.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.91,  -2.60,  -1.45>, < -3.43,  -2.57,  -1.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.96,  -2.53,  -1.70>, < -3.43,  -2.57,  -1.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.91,  -2.60,  -1.45>, < -4.11,  -2.54,  -1.90>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.32,  -2.47,  -2.35>, < -4.11,  -2.54,  -1.90>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.69,   2.50,  -5.84>, < -1.54,   2.13,  -5.58>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.40,   1.75,  -5.32>, < -1.54,   2.13,  -5.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.69,   2.50,  -5.84>, < -2.06,   2.33,  -6.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.43,   2.16,  -6.28>, < -2.06,   2.33,  -6.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.81,  -3.97,  -4.82>, < -1.58,  -4.07,  -5.24>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.34,  -4.16,  -5.65>, < -1.58,  -4.07,  -5.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.81,  -3.97,  -4.82>, < -1.50,  -3.94,  -4.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.18,  -3.91,  -4.11>, < -1.50,  -3.94,  -4.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.93,  -1.31,  -7.04>, < -1.40,  -1.45,  -6.95>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.87,  -1.60,  -6.87>, < -1.40,  -1.45,  -6.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.93,  -1.31,  -7.04>, < -0.82,  -1.06,  -6.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.71,  -0.82,  -6.18>, < -0.82,  -1.06,  -6.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.16,  -0.59,  -0.92>, <  0.76,  -0.78,  -1.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -0.98,  -1.32>, <  0.76,  -0.78,  -1.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.16,  -0.59,  -0.92>, <  1.03,  -0.53,  -0.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.91,  -0.48,   0.00>, <  1.03,  -0.53,  -0.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.05,  -2.17,  -1.75>, < -1.05,  -1.85,  -2.15>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.05,  -1.54,  -2.54>, < -1.05,  -1.85,  -2.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.05,  -2.17,  -1.75>, < -0.89,  -2.60,  -1.92>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.73,  -3.03,  -2.09>, < -0.89,  -2.60,  -1.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.63,   1.99,  -3.75>, <  1.30,   1.64,  -3.74>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.97,   1.28,  -3.74>, <  1.30,   1.64,  -3.74>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.63,   1.99,  -3.75>, <  1.44,   2.36,  -3.98>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.25,   2.73,  -4.21>, <  1.44,   2.36,  -3.98>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.59,   2.19,  -1.43>, < -3.73,   2.30,  -0.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.87,   2.42,  -0.50>, < -3.73,   2.30,  -0.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.59,   2.19,  -1.43>, < -3.12,   2.31,  -1.40>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.66,   2.43,  -1.37>, < -3.12,   2.31,  -1.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
