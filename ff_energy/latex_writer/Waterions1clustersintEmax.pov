#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -7.77*x up 7.55*y
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
atom(< -3.20,  -1.83,  -6.63>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -2.62,  -1.04,  -6.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -3.53,  -1.79,  -7.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -0.45,   2.47,  -3.33>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -1.02,   2.89,  -3.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  0.38,   2.59,  -3.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  0.41,   3.33,  -0.84>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(< -0.19,   3.10,  -1.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(< -0.05,   3.09,  -0.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -1.45,   0.60,  -5.92>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -0.77,   0.94,  -6.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -2.02,   1.36,  -5.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  1.31,  -1.79,  -5.62>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  2.01,  -2.49,  -5.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  0.65,  -2.29,  -6.21>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -1.66,  -0.36,  -1.52>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -1.83,  -0.63,  -0.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -2.12,   0.47,  -1.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  0.65,   0.36,   0.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  1.13,  -0.08,  -0.68>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -0.25,   0.31,  -0.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -0.94,  -2.56,  -3.54>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -1.71,  -2.52,  -2.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -1.06,  -3.47,  -3.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  1.94,  -1.19,  -2.15>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  2.83,  -1.24,  -1.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  2.00,  -1.96,  -2.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  3.44,  -0.37,  -6.41>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  3.36,   0.57,  -6.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  2.58,  -0.63,  -6.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -3.18,  -2.58,  -2.26>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -2.97,  -1.70,  -1.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -3.58,  -2.32,  -3.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  2.58,   1.78,  -1.86>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  1.90,   2.37,  -1.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  2.54,   1.91,  -2.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  0.05,  -0.12,  -3.79>, 0.81, rgb <0.50, 0.20, 0.39>, 0.0, jmol) // #36
atom(<  1.89,   2.18,  -5.20>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
cylinder {< -3.20,  -1.83,  -6.63>, < -2.91,  -1.44,  -6.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.62,  -1.04,  -6.51>, < -2.91,  -1.44,  -6.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.20,  -1.83,  -6.63>, < -3.37,  -1.81,  -7.08>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.53,  -1.79,  -7.52>, < -3.37,  -1.81,  -7.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.45,   2.47,  -3.33>, < -0.74,   2.68,  -3.64>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.02,   2.89,  -3.95>, < -0.74,   2.68,  -3.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.45,   2.47,  -3.33>, < -0.20,   1.18,  -3.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, < -0.20,   1.18,  -3.56>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.45,   2.47,  -3.33>, < -0.04,   2.53,  -3.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.38,   2.59,  -3.81>, < -0.04,   2.53,  -3.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.38,   2.59,  -3.81>, <  0.21,   1.24,  -3.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, <  0.21,   1.24,  -3.80>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.41,   3.33,  -0.84>, <  0.11,   3.22,  -1.22>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.19,   3.10,  -1.60>, <  0.11,   3.22,  -1.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.41,   3.33,  -0.84>, <  0.18,   3.21,  -0.43>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.05,   3.09,  -0.02>, <  0.18,   3.21,  -0.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.45,   0.60,  -5.92>, < -1.73,   0.98,  -5.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.02,   1.36,  -5.80>, < -1.73,   0.98,  -5.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.45,   0.60,  -5.92>, < -1.11,   0.77,  -6.26>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.77,   0.94,  -6.61>, < -1.11,   0.77,  -6.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.45,   0.60,  -5.92>, < -0.70,   0.24,  -4.85>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, < -0.70,   0.24,  -4.85>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.77,   0.94,  -6.61>, < -0.36,   0.41,  -5.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, < -0.36,   0.41,  -5.20>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -1.79,  -5.62>, <  0.68,  -0.95,  -4.71>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, <  0.68,  -0.95,  -4.71>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -1.79,  -5.62>, <  0.98,  -2.04,  -5.92>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.65,  -2.29,  -6.21>, <  0.98,  -2.04,  -5.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -1.79,  -5.62>, <  1.66,  -2.14,  -5.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.01,  -2.49,  -5.60>, <  1.66,  -2.14,  -5.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.66,  -0.36,  -1.52>, < -0.80,  -0.24,  -2.65>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, < -0.80,  -0.24,  -2.65>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -1.66,  -0.36,  -1.52>, < -1.74,  -0.50,  -1.05>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.83,  -0.63,  -0.57>, < -1.74,  -0.50,  -1.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.66,  -0.36,  -1.52>, < -1.89,   0.05,  -1.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.12,   0.47,  -1.57>, < -1.89,   0.05,  -1.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.12,   0.47,  -1.57>, < -1.03,   0.18,  -2.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, < -1.03,   0.18,  -2.68>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.65,   0.36,   0.00>, <  0.20,   0.34,  -0.14>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.25,   0.31,  -0.28>, <  0.20,   0.34,  -0.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.65,   0.36,   0.00>, <  0.89,   0.14,  -0.34>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.13,  -0.08,  -0.68>, <  0.89,   0.14,  -0.34>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,  -2.56,  -3.54>, < -1.00,  -3.02,  -3.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.06,  -3.47,  -3.81>, < -1.00,  -3.02,  -3.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,  -2.56,  -3.54>, < -0.44,  -1.34,  -3.66>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, < -0.44,  -1.34,  -3.66>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,  -2.56,  -3.54>, < -1.32,  -2.54,  -3.20>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.71,  -2.52,  -2.87>, < -1.32,  -2.54,  -3.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.71,  -2.52,  -2.87>, < -0.83,  -1.32,  -3.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, < -0.83,  -1.32,  -3.33>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -1.71,  -2.52,  -2.87>, < -2.44,  -2.55,  -2.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.18,  -2.58,  -2.26>, < -2.44,  -2.55,  -2.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.94,  -1.19,  -2.15>, <  0.99,  -0.66,  -2.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, <  0.99,  -0.66,  -2.97>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.94,  -1.19,  -2.15>, <  1.97,  -1.58,  -2.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.00,  -1.96,  -2.79>, <  1.97,  -1.58,  -2.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.94,  -1.19,  -2.15>, <  2.38,  -1.22,  -2.00>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.83,  -1.24,  -1.85>, <  2.38,  -1.22,  -2.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.00,  -1.96,  -2.79>, <  1.03,  -1.04,  -3.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, <  1.03,  -1.04,  -3.29>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  3.44,  -0.37,  -6.41>, <  3.40,   0.10,  -6.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.36,   0.57,  -6.17>, <  3.40,   0.10,  -6.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.44,  -0.37,  -6.41>, <  3.01,  -0.50,  -6.21>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,  -0.63,  -6.02>, <  3.01,  -0.50,  -6.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.18,  -2.58,  -2.26>, < -3.38,  -2.45,  -2.69>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.58,  -2.32,  -3.12>, < -3.38,  -2.45,  -2.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.18,  -2.58,  -2.26>, < -3.07,  -2.14,  -2.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.97,  -1.70,  -1.86>, < -3.07,  -2.14,  -2.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,   1.78,  -1.86>, <  2.56,   1.85,  -2.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.54,   1.91,  -2.84>, <  2.56,   1.85,  -2.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,   1.78,  -1.86>, <  2.24,   2.07,  -1.69>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.90,   2.37,  -1.53>, <  2.24,   2.07,  -1.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -0.12,  -3.79>, <  0.97,   1.03,  -4.49>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.89,   2.18,  -5.20>, <  0.97,   1.03,  -4.49>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
// no constraints
