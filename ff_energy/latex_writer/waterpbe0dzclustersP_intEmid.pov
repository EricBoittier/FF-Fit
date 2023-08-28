#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -12.28*x up 11.53*y
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
atom(< -0.19,   0.14,  -4.63>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  0.06,  -0.84,  -4.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -1.17,   0.03,  -4.34>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -5.36,   2.27,  -3.66>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -4.74,   2.74,  -4.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -5.72,   3.07,  -3.20>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(< -2.94,   0.10,  -3.63>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(< -3.30,   0.44,  -2.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(< -3.64,  -0.34,  -4.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  0.54,   5.06,  -5.04>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -0.31,   5.37,  -5.45>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  0.31,   4.16,  -4.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(< -4.66,   0.70,  -1.63>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -5.20,  -0.10,  -1.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -5.12,   1.14,  -2.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  0.23,  -4.49,  -6.65>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  1.00,  -4.60,  -6.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(<  0.05,  -3.50,  -6.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  0.03,  -1.25,  -0.11>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  0.69,  -1.58,  -0.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -0.74,  -1.74,  -0.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -3.20,  -5.23,  -5.71>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -2.74,  -5.11,  -6.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -2.64,  -4.62,  -5.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -2.66,   3.52,  -3.60>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(< -2.15,   2.89,  -3.05>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(< -2.70,   3.06,  -4.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  1.51,   2.56,  -6.77>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  1.06,   2.45,  -5.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  2.35,   2.76,  -6.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  3.63,  -1.98,  -5.04>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(<  2.69,  -2.08,  -4.78>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  3.72,  -2.62,  -5.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(< -2.90,   1.84,  -5.83>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -2.84,   1.10,  -5.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -3.01,   1.34,  -6.67>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  0.07,   1.84,  -8.85>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(<  0.88,   1.75,  -9.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(<  0.55,   2.14,  -7.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -5.04,  -3.67,  -4.17>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(< -4.49,  -4.18,  -4.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(< -4.96,  -2.78,  -4.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(<  5.08,   1.14,  -3.79>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(<  5.72,   0.56,  -4.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  4.49,   0.43,  -3.55>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(<  1.51,   2.80,  -0.91>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(<  1.88,   2.38,  -0.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(<  1.85,   3.73,  -0.78>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(<  1.94,  -2.02,  -1.86>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(<  2.33,  -2.95,  -1.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(<  2.68,  -1.37,  -1.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(<  1.07,  -2.60,  -4.29>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(<  0.71,  -3.44,  -3.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(<  1.24,  -2.10,  -3.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  0.15,  -1.34,  -7.16>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(<  1.04,  -1.12,  -7.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(<  0.21,  -0.84,  -6.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(< -2.22,  -3.15,  -3.82>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(< -2.23,  -2.44,  -3.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -2.06,  -3.94,  -3.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {< -0.19,   0.14,  -4.63>, < -0.68,   0.09,  -4.49>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.17,   0.03,  -4.34>, < -0.68,   0.09,  -4.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.19,   0.14,  -4.63>, < -0.07,  -0.35,  -4.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.06,  -0.84,  -4.36>, < -0.07,  -0.35,  -4.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.36,   2.27,  -3.66>, < -5.05,   2.50,  -3.95>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.74,   2.74,  -4.23>, < -5.05,   2.50,  -3.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.36,   2.27,  -3.66>, < -5.54,   2.67,  -3.43>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.72,   3.07,  -3.20>, < -5.54,   2.67,  -3.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.94,   0.10,  -3.63>, < -3.12,   0.27,  -3.24>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.30,   0.44,  -2.84>, < -3.12,   0.27,  -3.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.94,   0.10,  -3.63>, < -3.29,  -0.12,  -3.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.64,  -0.34,  -4.09>, < -3.29,  -0.12,  -3.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.54,   5.06,  -5.04>, <  0.11,   5.22,  -5.25>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.31,   5.37,  -5.45>, <  0.11,   5.22,  -5.25>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.54,   5.06,  -5.04>, <  0.42,   4.61,  -4.89>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.31,   4.16,  -4.75>, <  0.42,   4.61,  -4.89>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.66,   0.70,  -1.63>, < -4.89,   0.92,  -2.02>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.12,   1.14,  -2.41>, < -4.89,   0.92,  -2.02>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.66,   0.70,  -1.63>, < -4.93,   0.30,  -1.53>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.20,  -0.10,  -1.43>, < -4.93,   0.30,  -1.53>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.23,  -4.49,  -6.65>, <  0.14,  -4.00,  -6.62>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.50,  -6.60>, <  0.14,  -4.00,  -6.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.23,  -4.49,  -6.65>, <  0.62,  -4.54,  -6.33>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.00,  -4.60,  -6.00>, <  0.62,  -4.54,  -6.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -1.25,  -0.11>, < -0.36,  -1.49,  -0.31>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.74,  -1.74,  -0.51>, < -0.36,  -1.49,  -0.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -1.25,  -0.11>, <  0.36,  -1.41,  -0.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.69,  -1.58,  -0.71>, <  0.36,  -1.41,  -0.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.20,  -5.23,  -5.71>, < -2.97,  -5.17,  -6.14>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.74,  -5.11,  -6.58>, < -2.97,  -5.17,  -6.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.20,  -5.23,  -5.71>, < -2.92,  -4.93,  -5.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.64,  -4.62,  -5.10>, < -2.92,  -4.93,  -5.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.66,   3.52,  -3.60>, < -2.41,   3.20,  -3.32>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.15,   2.89,  -3.05>, < -2.41,   3.20,  -3.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.66,   3.52,  -3.60>, < -2.68,   3.29,  -4.03>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.70,   3.06,  -4.47>, < -2.68,   3.29,  -4.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,   2.56,  -6.77>, <  1.29,   2.51,  -6.32>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.06,   2.45,  -5.87>, <  1.29,   2.51,  -6.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,   2.56,  -6.77>, <  1.03,   2.35,  -7.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.55,   2.14,  -7.97>, <  1.03,   2.35,  -7.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,   2.56,  -6.77>, <  1.93,   2.66,  -6.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.35,   2.76,  -6.36>, <  1.93,   2.66,  -6.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.63,  -1.98,  -5.04>, <  3.68,  -2.30,  -5.39>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.72,  -2.62,  -5.74>, <  3.68,  -2.30,  -5.39>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.63,  -1.98,  -5.04>, <  3.16,  -2.03,  -4.91>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.69,  -2.08,  -4.78>, <  3.16,  -2.03,  -4.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.90,   1.84,  -5.83>, < -2.95,   1.59,  -6.25>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.01,   1.34,  -6.67>, < -2.95,   1.59,  -6.25>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.90,   1.84,  -5.83>, < -2.87,   1.47,  -5.53>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.84,   1.10,  -5.23>, < -2.87,   1.47,  -5.53>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.07,   1.84,  -8.85>, <  0.47,   1.79,  -9.16>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.88,   1.75,  -9.47>, <  0.47,   1.79,  -9.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.07,   1.84,  -8.85>, <  0.31,   1.99,  -8.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.55,   2.14,  -7.97>, <  0.31,   1.99,  -8.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.04,  -3.67,  -4.17>, < -5.00,  -3.23,  -4.36>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.96,  -2.78,  -4.54>, < -5.00,  -3.23,  -4.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.04,  -3.67,  -4.17>, < -4.76,  -3.92,  -4.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.49,  -4.18,  -4.79>, < -4.76,  -3.92,  -4.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.08,   1.14,  -3.79>, <  4.78,   0.78,  -3.67>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.49,   0.43,  -3.55>, <  4.78,   0.78,  -3.67>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.08,   1.14,  -3.79>, <  5.40,   0.85,  -4.03>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.72,   0.56,  -4.27>, <  5.40,   0.85,  -4.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,   2.80,  -0.91>, <  1.68,   3.26,  -0.84>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.85,   3.73,  -0.78>, <  1.68,   3.26,  -0.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,   2.80,  -0.91>, <  1.69,   2.59,  -0.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.88,   2.38,  -0.03>, <  1.69,   2.59,  -0.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.94,  -2.02,  -1.86>, <  2.31,  -1.70,  -1.78>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.68,  -1.37,  -1.70>, <  2.31,  -1.70,  -1.78>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.94,  -2.02,  -1.86>, <  2.14,  -2.49,  -1.92>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.33,  -2.95,  -1.97>, <  2.14,  -2.49,  -1.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.07,  -2.60,  -4.29>, <  0.89,  -3.02,  -4.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.71,  -3.44,  -3.83>, <  0.89,  -3.02,  -4.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.07,  -2.60,  -4.29>, <  1.16,  -2.35,  -3.88>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.24,  -2.10,  -3.46>, <  1.16,  -2.35,  -3.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.15,  -1.34,  -7.16>, <  0.18,  -1.09,  -6.74>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.21,  -0.84,  -6.32>, <  0.18,  -1.09,  -6.74>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.15,  -1.34,  -7.16>, <  0.59,  -1.23,  -7.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.04,  -1.12,  -7.54>, <  0.59,  -1.23,  -7.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.22,  -3.15,  -3.82>, < -2.23,  -2.79,  -3.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.23,  -2.44,  -3.11>, < -2.23,  -2.79,  -3.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.22,  -3.15,  -3.82>, < -2.14,  -3.54,  -3.49>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.06,  -3.94,  -3.17>, < -2.14,  -3.54,  -3.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
