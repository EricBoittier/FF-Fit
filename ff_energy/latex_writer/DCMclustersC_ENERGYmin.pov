#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -20.20*x up 19.64*y
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
atom(<  0.20,   5.46,  -7.23>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(<  0.51,   5.08,  -5.50>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(<  1.57,   6.45,  -7.85>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(< -0.75,   6.01,  -7.34>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(<  0.14,   4.48,  -7.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  2.49,  -0.33, -11.88>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(<  1.65,  -0.42, -10.28>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(<  1.60,   0.91, -12.84>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(<  2.38,  -1.33, -12.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  3.56,   0.04, -11.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(<  6.64,  -1.73, -12.48>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(<  6.53,  -0.23, -11.40>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(<  5.04,  -2.00, -13.16>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(<  6.97,  -2.61, -11.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  7.34,  -1.46, -13.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -0.39,   4.01, -20.70>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< -0.79,   4.69, -19.09>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(< -0.31,   2.24, -20.61>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(<  0.60,   4.35, -21.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(< -1.23,   4.21, -21.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  7.64,   1.52, -18.47>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(<  8.29,   0.21, -17.50>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(<  8.91,   2.18, -19.47>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(<  7.25,   2.32, -17.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  6.85,   1.15, -19.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(< -0.86,   5.03, -12.77>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(< -0.32,   5.45, -11.16>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(< -2.61,   4.88, -12.57>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(< -0.38,   4.05, -12.99>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -0.64,   5.76, -13.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -1.93,  -5.40,  -6.67>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(< -2.77,  -5.81,  -5.14>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(< -1.02,  -3.84,  -6.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(< -2.76,  -5.35,  -7.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(< -1.22,  -6.20,  -6.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -4.22,  -3.10, -14.91>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(< -3.06,  -1.79, -15.23>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(< -3.91,  -4.18, -16.20>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(< -5.31,  -2.81, -14.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -4.02,  -3.52, -13.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(<  8.74,   4.95,  -7.11>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(<  7.54,   3.97,  -8.05>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(<  7.60,   6.08,  -6.38>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(<  9.49,   5.43,  -7.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  9.21,   4.36,  -6.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(<  3.69,   4.40, -10.43>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(<  4.33,   5.00,  -8.90>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(<  4.81,   5.19, -11.61>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(<  2.63,   4.76, -10.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(<  3.83,   3.28, -10.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(<  8.03,   4.58, -15.52>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(<  6.59,   3.76, -15.25>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(<  7.62,   6.19, -14.95>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(<  8.24,   4.58, -16.62>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  8.76,   4.12, -14.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(< -7.51,   2.03, -17.18>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(< -6.57,   3.56, -17.43>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(< -8.42,   2.36, -15.68>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(< -6.76,   1.23, -16.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -8.20,   1.82, -18.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(<  3.07,   6.85,  -2.13>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(<  3.37,   6.62,  -3.81>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(<  3.62,   8.45,  -1.79>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(<  3.57,   6.10,  -1.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(<  1.96,   6.78,  -1.98>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(< -6.64,   7.32, -11.62>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(< -6.14,   8.94, -12.12>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(< -8.17,   7.53, -10.78>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(< -5.87,   6.91, -10.93>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(< -6.73,   6.71, -12.55>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(<  6.62,  -1.34,  -7.46>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(<  8.39,  -1.43,  -7.81>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(<  5.89,  -2.95,  -7.14>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(<  6.06,  -0.79,  -8.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(<  6.53,  -0.81,  -6.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(<  0.09,  -4.30, -14.16>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(<  0.34,  -3.05, -12.93>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(<  1.43,  -4.32, -15.36>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(< -0.81,  -4.05, -14.78>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(< -0.03,  -5.34, -13.78>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(< -8.01,   1.57, -12.57>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(< -9.21,   2.79, -11.99>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(< -6.92,   1.09, -11.19>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(< -7.34,   1.99, -13.33>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(< -8.57,   0.71, -12.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(< -4.02,   1.85, -21.20>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(< -3.57,   2.57, -19.59>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(< -5.64,   1.07, -20.94>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(< -3.27,   1.04, -21.55>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(< -4.08,   2.64, -22.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(< -3.88,   7.10,  -0.62>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(< -3.37,   7.22,  -2.29>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(< -5.15,   5.88,  -0.38>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(< -4.32,   8.05,  -0.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(< -3.01,   6.81,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(< -1.66,  -5.89, -10.16>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(< -3.36,  -5.32, -10.55>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(< -1.23,  -7.33, -11.13>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(< -1.54,  -6.23,  -9.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(< -0.90,  -5.04, -10.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {<  0.20,   5.46,  -7.23>, <  0.17,   4.97,  -7.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.14,   4.48,  -7.74>, <  0.17,   4.97,  -7.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.20,   5.46,  -7.23>, <  0.35,   5.27,  -6.36>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.51,   5.08,  -5.50>, <  0.35,   5.27,  -6.36>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.20,   5.46,  -7.23>, < -0.28,   5.74,  -7.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.75,   6.01,  -7.34>, < -0.28,   5.74,  -7.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.20,   5.46,  -7.23>, <  0.88,   5.95,  -7.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.57,   6.45,  -7.85>, <  0.88,   5.95,  -7.54>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.49,  -0.33, -11.88>, <  3.02,  -0.14, -11.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.56,   0.04, -11.77>, <  3.02,  -0.14, -11.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.49,  -0.33, -11.88>, <  2.07,  -0.37, -11.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.65,  -0.42, -10.28>, <  2.07,  -0.37, -11.08>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.49,  -0.33, -11.88>, <  2.04,   0.29, -12.36>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.60,   0.91, -12.84>, <  2.04,   0.29, -12.36>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.49,  -0.33, -11.88>, <  2.44,  -0.83, -12.12>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.38,  -1.33, -12.35>, <  2.44,  -0.83, -12.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.64,  -1.73, -12.48>, <  6.81,  -2.17, -12.16>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.97,  -2.61, -11.83>, <  6.81,  -2.17, -12.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.64,  -1.73, -12.48>, <  6.59,  -0.98, -11.94>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.53,  -0.23, -11.40>, <  6.59,  -0.98, -11.94>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.64,  -1.73, -12.48>, <  5.84,  -1.86, -12.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.04,  -2.00, -13.16>, <  5.84,  -1.86, -12.82>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.64,  -1.73, -12.48>, <  6.99,  -1.60, -12.90>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.34,  -1.46, -13.32>, <  6.99,  -1.60, -12.90>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.39,   4.01, -20.70>, < -0.81,   4.11, -21.03>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.23,   4.21, -21.36>, < -0.81,   4.11, -21.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.39,   4.01, -20.70>, < -0.35,   3.13, -20.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.31,   2.24, -20.61>, < -0.35,   3.13, -20.66>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.39,   4.01, -20.70>, <  0.11,   4.18, -20.92>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.60,   4.35, -21.13>, <  0.11,   4.18, -20.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.39,   4.01, -20.70>, < -0.59,   4.35, -19.90>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.79,   4.69, -19.09>, < -0.59,   4.35, -19.90>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.64,   1.52, -18.47>, <  7.24,   1.33, -18.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.85,   1.15, -19.16>, <  7.24,   1.33, -18.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.64,   1.52, -18.47>, <  7.96,   0.86, -17.99>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.29,   0.21, -17.50>, <  7.96,   0.86, -17.99>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.64,   1.52, -18.47>, <  8.27,   1.85, -18.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.91,   2.18, -19.47>, <  8.27,   1.85, -18.97>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.64,   1.52, -18.47>, <  7.44,   1.92, -18.14>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.25,   2.32, -17.81>, <  7.44,   1.92, -18.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.86,   5.03, -12.77>, < -0.59,   5.24, -11.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.32,   5.45, -11.16>, < -0.59,   5.24, -11.97>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.86,   5.03, -12.77>, < -1.73,   4.95, -12.67>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.61,   4.88, -12.57>, < -1.73,   4.95, -12.67>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.86,   5.03, -12.77>, < -0.62,   4.54, -12.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.38,   4.05, -12.99>, < -0.62,   4.54, -12.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.86,   5.03, -12.77>, < -0.75,   5.39, -13.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.64,   5.76, -13.57>, < -0.75,   5.39, -13.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.93,  -5.40,  -6.67>, < -2.35,  -5.60,  -5.91>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.77,  -5.81,  -5.14>, < -2.35,  -5.60,  -5.91>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.93,  -5.40,  -6.67>, < -1.57,  -5.80,  -6.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.22,  -6.20,  -6.92>, < -1.57,  -5.80,  -6.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.93,  -5.40,  -6.67>, < -1.47,  -4.62,  -6.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.02,  -3.84,  -6.45>, < -1.47,  -4.62,  -6.56>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.93,  -5.40,  -6.67>, < -2.34,  -5.37,  -7.04>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.76,  -5.35,  -7.41>, < -2.34,  -5.37,  -7.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.22,  -3.10, -14.91>, < -3.64,  -2.45, -15.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.06,  -1.79, -15.23>, < -3.64,  -2.45, -15.07>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.22,  -3.10, -14.91>, < -4.07,  -3.64, -15.55>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.91,  -4.18, -16.20>, < -4.07,  -3.64, -15.55>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.22,  -3.10, -14.91>, < -4.76,  -2.95, -14.94>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.31,  -2.81, -14.97>, < -4.76,  -2.95, -14.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.22,  -3.10, -14.91>, < -4.12,  -3.31, -14.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.02,  -3.52, -13.92>, < -4.12,  -3.31, -14.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.74,   4.95,  -7.11>, <  8.14,   4.46,  -7.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.54,   3.97,  -8.05>, <  8.14,   4.46,  -7.58>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  8.74,   4.95,  -7.11>, <  8.98,   4.66,  -6.75>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  9.21,   4.36,  -6.38>, <  8.98,   4.66,  -6.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.74,   4.95,  -7.11>, <  9.12,   5.19,  -7.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  9.49,   5.43,  -7.75>, <  9.12,   5.19,  -7.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.74,   4.95,  -7.11>, <  8.17,   5.52,  -6.75>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.60,   6.08,  -6.38>, <  8.17,   5.52,  -6.75>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.69,   4.40, -10.43>, <  3.76,   3.84, -10.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,   3.28, -10.37>, <  3.76,   3.84, -10.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.69,   4.40, -10.43>, <  3.16,   4.58, -10.52>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.63,   4.76, -10.60>, <  3.16,   4.58, -10.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.69,   4.40, -10.43>, <  4.01,   4.70,  -9.67>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.33,   5.00,  -8.90>, <  4.01,   4.70,  -9.67>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.69,   4.40, -10.43>, <  4.25,   4.79, -11.02>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.81,   5.19, -11.61>, <  4.25,   4.79, -11.02>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  8.03,   4.58, -15.52>, <  7.31,   4.17, -15.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.59,   3.76, -15.25>, <  7.31,   4.17, -15.38>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  8.03,   4.58, -15.52>, <  8.13,   4.58, -16.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.24,   4.58, -16.62>, <  8.13,   4.58, -16.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.03,   4.58, -15.52>, <  7.82,   5.38, -15.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.62,   6.19, -14.95>, <  7.82,   5.38, -15.24>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  8.03,   4.58, -15.52>, <  8.39,   4.35, -15.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.76,   4.12, -14.86>, <  8.39,   4.35, -15.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.51,   2.03, -17.18>, < -7.13,   1.63, -17.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.76,   1.23, -16.97>, < -7.13,   1.63, -17.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.51,   2.03, -17.18>, < -7.85,   1.93, -17.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.20,   1.82, -18.02>, < -7.85,   1.93, -17.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.51,   2.03, -17.18>, < -7.97,   2.20, -16.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.42,   2.36, -15.68>, < -7.97,   2.20, -16.43>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.51,   2.03, -17.18>, < -7.04,   2.79, -17.30>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.57,   3.56, -17.43>, < -7.04,   2.79, -17.30>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.07,   6.85,  -2.13>, <  2.51,   6.82,  -2.05>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.96,   6.78,  -1.98>, <  2.51,   6.82,  -2.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.07,   6.85,  -2.13>, <  3.32,   6.47,  -1.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.57,   6.10,  -1.52>, <  3.32,   6.47,  -1.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.07,   6.85,  -2.13>, <  3.35,   7.65,  -1.96>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.62,   8.45,  -1.79>, <  3.35,   7.65,  -1.96>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.07,   6.85,  -2.13>, <  3.22,   6.74,  -2.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.37,   6.62,  -3.81>, <  3.22,   6.74,  -2.97>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.64,   7.32, -11.62>, < -7.40,   7.42, -11.20>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.17,   7.53, -10.78>, < -7.40,   7.42, -11.20>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.64,   7.32, -11.62>, < -6.26,   7.11, -11.28>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.87,   6.91, -10.93>, < -6.26,   7.11, -11.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.64,   7.32, -11.62>, < -6.68,   7.01, -12.09>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.73,   6.71, -12.55>, < -6.68,   7.01, -12.09>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.64,   7.32, -11.62>, < -6.39,   8.13, -11.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.14,   8.94, -12.12>, < -6.39,   8.13, -11.88>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.62,  -1.34,  -7.46>, <  6.58,  -1.08,  -6.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.53,  -0.81,  -6.47>, <  6.58,  -1.08,  -6.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.62,  -1.34,  -7.46>, <  6.34,  -1.07,  -7.86>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.06,  -0.79,  -8.26>, <  6.34,  -1.07,  -7.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.62,  -1.34,  -7.46>, <  6.25,  -2.14,  -7.30>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.89,  -2.95,  -7.14>, <  6.25,  -2.14,  -7.30>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.62,  -1.34,  -7.46>, <  7.51,  -1.38,  -7.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.39,  -1.43,  -7.81>, <  7.51,  -1.38,  -7.64>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.09,  -4.30, -14.16>, < -0.36,  -4.18, -14.47>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.81,  -4.05, -14.78>, < -0.36,  -4.18, -14.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.09,  -4.30, -14.16>, <  0.76,  -4.31, -14.76>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.43,  -4.32, -15.36>, <  0.76,  -4.31, -14.76>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.09,  -4.30, -14.16>, <  0.21,  -3.68, -13.55>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.34,  -3.05, -12.93>, <  0.21,  -3.68, -13.55>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.09,  -4.30, -14.16>, <  0.03,  -4.82, -13.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.03,  -5.34, -13.78>, <  0.03,  -4.82, -13.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -8.01,   1.57, -12.57>, < -7.46,   1.33, -11.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.92,   1.09, -11.19>, < -7.46,   1.33, -11.88>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -8.01,   1.57, -12.57>, < -8.29,   1.14, -12.76>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.57,   0.71, -12.95>, < -8.29,   1.14, -12.76>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -8.01,   1.57, -12.57>, < -7.67,   1.78, -12.95>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.34,   1.99, -13.33>, < -7.67,   1.78, -12.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -8.01,   1.57, -12.57>, < -8.61,   2.18, -12.28>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -9.21,   2.79, -11.99>, < -8.61,   2.18, -12.28>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.02,   1.85, -21.20>, < -3.65,   1.44, -21.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.27,   1.04, -21.55>, < -3.65,   1.44, -21.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.02,   1.85, -21.20>, < -4.83,   1.46, -21.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.64,   1.07, -20.94>, < -4.83,   1.46, -21.07>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.02,   1.85, -21.20>, < -3.80,   2.21, -20.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.57,   2.57, -19.59>, < -3.80,   2.21, -20.40>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.02,   1.85, -21.20>, < -4.05,   2.24, -21.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.08,   2.64, -22.00>, < -4.05,   2.24, -21.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.88,   7.10,  -0.62>, < -4.51,   6.49,  -0.50>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.15,   5.88,  -0.38>, < -4.51,   6.49,  -0.50>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.88,   7.10,  -0.62>, < -3.62,   7.16,  -1.45>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.37,   7.22,  -2.29>, < -3.62,   7.16,  -1.45>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.88,   7.10,  -0.62>, < -4.10,   7.58,  -0.47>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.32,   8.05,  -0.31>, < -4.10,   7.58,  -0.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.88,   7.10,  -0.62>, < -3.44,   6.96,  -0.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.01,   6.81,   0.00>, < -3.44,   6.96,  -0.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.66,  -5.89, -10.16>, < -1.28,  -5.46, -10.20>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.90,  -5.04, -10.25>, < -1.28,  -5.46, -10.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.66,  -5.89, -10.16>, < -2.51,  -5.60, -10.35>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.36,  -5.32, -10.55>, < -2.51,  -5.60, -10.35>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.66,  -5.89, -10.16>, < -1.45,  -6.61, -10.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.23,  -7.33, -11.13>, < -1.45,  -6.61, -10.65>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.66,  -5.89, -10.16>, < -1.60,  -6.06,  -9.61>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.54,  -6.23,  -9.07>, < -1.60,  -6.06,  -9.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
