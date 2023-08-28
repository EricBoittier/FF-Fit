#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -8.96*x up 16.10*y
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
atom(<  1.80,  -2.25,  -1.47>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  2.33,  -2.06,  -0.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(<  1.94,  -1.42,  -1.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -1.40,  -6.99,  -4.62>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -0.76,  -7.54,  -4.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -1.00,  -7.04,  -5.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  1.82,  -3.47,  -4.33>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  2.21,  -2.77,  -4.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  2.13,  -4.23,  -4.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  3.54,  -4.53,  -1.63>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  4.14,  -3.80,  -1.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  2.74,  -4.05,  -1.94>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  0.47,  -5.44,  -1.58>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  0.62,  -6.37,  -1.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  1.21,  -5.29,  -0.99>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  0.50,  -0.35,  -5.52>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  1.47,  -0.20,  -5.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(<  0.13,  -0.38,  -6.39>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(< -2.97,  -4.37,  -2.27>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(< -3.23,  -4.81,  -3.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -3.42,  -3.49,  -2.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -0.90,  -2.38,  -0.39>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -1.75,  -2.06,  -0.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -0.18,  -1.95,  -0.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -3.77,  -5.80,  -4.41>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(< -2.81,  -6.15,  -4.39>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(< -3.92,  -5.34,  -5.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(< -3.34,  -1.55,  -1.67>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(< -3.40,  -0.83,  -2.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -4.14,  -1.44,  -1.15>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -0.57,  -1.11,  -3.12>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -1.10,  -0.31,  -2.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  0.15,  -0.85,  -3.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(< -2.54,  -5.96,   0.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -1.59,  -6.12,  -0.21>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -2.88,  -5.36,  -0.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(< -2.08,  -3.46,  -5.74>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(< -3.04,  -3.29,  -5.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(< -1.89,  -3.65,  -6.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -0.56,  -3.66,  -3.46>, 0.81, rgb <0.50, 0.20, 0.39>, 0.0, jmol) // #39
cylinder {<  1.80,  -2.25,  -1.47>, <  0.62,  -2.95,  -2.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, <  0.62,  -2.95,  -2.46>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.80,  -2.25,  -1.47>, <  1.87,  -1.83,  -1.70>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.94,  -1.42,  -1.92>, <  1.87,  -1.83,  -1.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.80,  -2.25,  -1.47>, <  2.06,  -2.16,  -1.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.33,  -2.06,  -0.66>, <  2.06,  -2.16,  -1.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.40,  -6.99,  -4.62>, < -2.11,  -6.57,  -4.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.81,  -6.15,  -4.39>, < -2.11,  -6.57,  -4.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.40,  -6.99,  -4.62>, < -1.20,  -7.02,  -5.08>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.00,  -7.04,  -5.54>, < -1.20,  -7.02,  -5.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.40,  -6.99,  -4.62>, < -1.08,  -7.27,  -4.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.76,  -7.54,  -4.12>, < -1.08,  -7.27,  -4.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.82,  -3.47,  -4.33>, <  1.98,  -3.85,  -4.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.13,  -4.23,  -4.82>, <  1.98,  -3.85,  -4.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.82,  -3.47,  -4.33>, <  0.63,  -3.56,  -3.89>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, <  0.63,  -3.56,  -3.89>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.82,  -3.47,  -4.33>, <  2.02,  -3.12,  -4.58>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.21,  -2.77,  -4.83>, <  2.02,  -3.12,  -4.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.13,  -4.23,  -4.82>, <  0.79,  -3.94,  -4.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, <  0.79,  -3.94,  -4.14>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  3.54,  -4.53,  -1.63>, <  3.84,  -4.16,  -1.53>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.14,  -3.80,  -1.43>, <  3.84,  -4.16,  -1.53>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.54,  -4.53,  -1.63>, <  3.14,  -4.29,  -1.79>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.74,  -4.05,  -1.94>, <  3.14,  -4.29,  -1.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.47,  -5.44,  -1.58>, <  0.54,  -5.91,  -1.65>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.62,  -6.37,  -1.71>, <  0.54,  -5.91,  -1.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.47,  -5.44,  -1.58>, <  0.84,  -5.37,  -1.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.21,  -5.29,  -0.99>, <  0.84,  -5.37,  -1.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.47,  -5.44,  -1.58>, < -0.04,  -4.55,  -2.52>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, < -0.04,  -4.55,  -2.52>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.50,  -0.35,  -5.52>, <  0.98,  -0.27,  -5.67>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.47,  -0.20,  -5.82>, <  0.98,  -0.27,  -5.67>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.50,  -0.35,  -5.52>, <  0.31,  -0.36,  -5.96>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.13,  -0.38,  -6.39>, <  0.31,  -0.36,  -5.96>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.97,  -4.37,  -2.27>, < -3.10,  -4.59,  -2.70>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.23,  -4.81,  -3.13>, < -3.10,  -4.59,  -2.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.97,  -4.37,  -2.27>, < -3.19,  -3.93,  -2.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.42,  -3.49,  -2.31>, < -3.19,  -3.93,  -2.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.97,  -4.37,  -2.27>, < -1.76,  -4.01,  -2.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, < -1.76,  -4.01,  -2.86>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -3.23,  -4.81,  -3.13>, < -1.89,  -4.23,  -3.30>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, < -1.89,  -4.23,  -3.30>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -3.42,  -3.49,  -2.31>, < -1.99,  -3.57,  -2.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, < -1.99,  -3.57,  -2.88>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.90,  -2.38,  -0.39>, < -1.32,  -2.22,  -0.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,  -2.06,  -0.83>, < -1.32,  -2.22,  -0.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.90,  -2.38,  -0.39>, < -0.73,  -3.02,  -1.93>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, < -0.73,  -3.02,  -1.93>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.90,  -2.38,  -0.39>, < -0.54,  -2.16,  -0.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.18,  -1.95,  -0.97>, < -0.54,  -2.16,  -0.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.18,  -1.95,  -0.97>, < -0.37,  -2.80,  -2.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, < -0.37,  -2.80,  -2.21>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -3.77,  -5.80,  -4.41>, < -3.29,  -5.98,  -4.40>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.81,  -6.15,  -4.39>, < -3.29,  -5.98,  -4.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.77,  -5.80,  -4.41>, < -3.85,  -5.57,  -4.83>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.92,  -5.34,  -5.26>, < -3.85,  -5.57,  -4.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.34,  -1.55,  -1.67>, < -3.74,  -1.49,  -1.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.14,  -1.44,  -1.15>, < -3.74,  -1.49,  -1.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.34,  -1.55,  -1.67>, < -3.37,  -1.19,  -1.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.40,  -0.83,  -2.27>, < -3.37,  -1.19,  -1.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.57,  -1.11,  -3.12>, < -0.83,  -0.71,  -3.04>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,  -0.31,  -2.96>, < -0.83,  -0.71,  -3.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.57,  -1.11,  -3.12>, < -0.56,  -2.38,  -3.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, < -0.56,  -2.38,  -3.29>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.57,  -1.11,  -3.12>, < -0.21,  -0.98,  -3.42>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.15,  -0.85,  -3.72>, < -0.21,  -0.98,  -3.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.15,  -0.85,  -3.72>, < -0.21,  -2.25,  -3.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, < -0.21,  -2.25,  -3.59>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -2.54,  -5.96,   0.00>, < -2.07,  -6.04,  -0.11>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.59,  -6.12,  -0.21>, < -2.07,  -6.04,  -0.11>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.54,  -5.96,   0.00>, < -2.71,  -5.66,  -0.36>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.88,  -5.36,  -0.72>, < -2.71,  -5.66,  -0.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.08,  -3.46,  -5.74>, < -1.98,  -3.55,  -6.22>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.89,  -3.65,  -6.69>, < -1.98,  -3.55,  -6.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.08,  -3.46,  -5.74>, < -2.56,  -3.37,  -5.79>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.04,  -3.29,  -5.84>, < -2.56,  -3.37,  -5.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.08,  -3.46,  -5.74>, < -1.32,  -3.56,  -4.60>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -3.66,  -3.46>, < -1.32,  -3.56,  -4.60>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
// no constraints
