#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -17.77*x up 17.92*y
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
atom(<  1.49,   1.03,  -9.74>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(<  0.71,   2.58, -10.24>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(<  1.45,  -0.16, -11.07>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(<  0.92,   0.65,  -8.89>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(<  2.56,   1.20,  -9.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -0.97,   1.88,  -5.17>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(< -1.41,   0.50,  -6.19>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(< -2.38,   2.70,  -4.46>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(< -0.46,   2.62,  -5.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -0.27,   1.57,  -4.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(< -7.20,  -2.79,  -9.83>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(< -8.06,  -3.11,  -8.29>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(< -5.41,  -2.77,  -9.50>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(< -7.57,  -1.84, -10.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -7.47,  -3.61, -10.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  0.30,   6.74,  -4.44>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< -0.32,   6.89,  -2.81>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(< -0.29,   8.13,  -5.44>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(< -0.16,   5.82,  -4.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(<  1.39,   6.69,  -4.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  7.40,   0.44,  -5.12>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(<  6.00,   1.52,  -5.63>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(<  7.39,  -0.97,  -6.23>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(<  7.31,   0.16,  -4.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  8.34,   1.01,  -5.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(<  4.94,  -4.02,  -3.52>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(<  4.10,  -2.70,  -2.63>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(<  6.22,  -3.58,  -4.64>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(<  4.16,  -4.63,  -4.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  5.39,  -4.65,  -2.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  2.11,   5.34,  -7.85>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(<  2.66,   4.97,  -6.16>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(<  3.61,   5.63,  -8.78>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(<  1.43,   6.18,  -7.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(<  1.61,   4.48,  -8.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -1.79,  -5.30, -12.08>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(< -2.54,  -4.51, -10.71>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(< -0.23,  -5.98, -11.58>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(< -1.71,  -4.55, -12.91>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -2.46,  -6.07, -12.39>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(< -3.53,  -2.88,  -5.30>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(< -4.66,  -3.77,  -6.32>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(< -1.93,  -2.88,  -6.11>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(< -3.49,  -3.40,  -4.34>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(< -3.88,  -1.86,  -5.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -2.70,   3.28,  -8.02>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(< -1.35,   4.37,  -7.58>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(< -3.76,   4.03,  -9.30>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(< -3.25,   3.06,  -7.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(< -2.25,   2.38,  -8.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -2.95,   0.13, -13.95>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(< -1.69,   0.80, -12.86>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(< -3.16,   1.09, -15.46>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(< -3.88,   0.06, -13.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -2.59,  -0.85, -14.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(<  4.08,   2.26, -16.48>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(<  5.63,   3.14, -16.76>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(<  4.27,   0.47, -16.79>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(<  3.31,   2.70, -17.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(<  3.70,   2.40, -15.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(<  5.97,   2.94,  -9.53>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(<  4.94,   2.27, -10.80>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(<  7.23,   1.75,  -9.12>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(<  5.38,   3.13,  -8.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(<  6.39,   3.84,  -9.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(<  2.59,   4.94,  -1.42>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(<  3.26,   4.18,  -2.89>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(<  2.88,   3.94,   0.00>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(<  1.50,   5.11,  -1.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(<  3.12,   5.89,  -1.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(<  3.26,  -4.38,  -7.14>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(<  1.90,  -3.42,  -6.53>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(<  3.89,  -3.73,  -8.71>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(<  2.92,  -5.43,  -7.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(<  4.06,  -4.33,  -6.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(<  4.67,  -1.88, -12.06>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(<  6.26,  -2.12, -12.95>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(<  3.35,  -2.98, -12.61>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(<  4.30,  -0.85, -12.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(<  4.91,  -2.00, -10.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(<  0.76,   5.33, -12.17>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(<  0.94,   6.69, -13.28>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(< -0.06,   5.88, -10.72>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(<  1.78,   4.98, -11.93>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(<  0.10,   4.60, -12.73>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(< -0.12,  -4.69,  -3.58>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(< -1.77,  -5.52,  -3.64>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(< -0.03,  -3.30,  -2.44>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(<  0.14,  -4.28,  -4.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(<  0.65,  -5.37,  -3.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(< -3.38,   5.67, -12.53>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(< -5.11,   5.92, -12.58>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(< -2.83,   4.13, -13.30>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(< -3.09,   5.71, -11.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(< -2.93,   6.50, -13.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(<  0.91,  -1.90, -16.23>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(<  0.14,  -3.39, -15.50>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(<  0.84,  -0.66, -14.92>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(<  0.35,  -1.51, -17.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(<  1.96,  -2.07, -16.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {<  1.49,   1.03,  -9.74>, <  1.47,   0.44, -10.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.45,  -0.16, -11.07>, <  1.47,   0.44, -10.41>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.49,   1.03,  -9.74>, <  1.21,   0.84,  -9.32>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.92,   0.65,  -8.89>, <  1.21,   0.84,  -9.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.49,   1.03,  -9.74>, <  2.02,   1.11,  -9.57>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.56,   1.20,  -9.41>, <  2.02,   1.11,  -9.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.49,   1.03,  -9.74>, <  1.10,   1.80,  -9.99>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.71,   2.58, -10.24>, <  1.10,   1.80,  -9.99>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.97,   1.88,  -5.17>, < -1.19,   1.19,  -5.68>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.41,   0.50,  -6.19>, < -1.19,   1.19,  -5.68>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.97,   1.88,  -5.17>, < -0.62,   1.73,  -4.77>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.27,   1.57,  -4.38>, < -0.62,   1.73,  -4.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.97,   1.88,  -5.17>, < -1.68,   2.29,  -4.81>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.38,   2.70,  -4.46>, < -1.68,   2.29,  -4.81>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.97,   1.88,  -5.17>, < -0.71,   2.25,  -5.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.46,   2.62,  -5.84>, < -0.71,   2.25,  -5.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.20,  -2.79,  -9.83>, < -6.31,  -2.78,  -9.67>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.41,  -2.77,  -9.50>, < -6.31,  -2.78,  -9.67>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.20,  -2.79,  -9.83>, < -7.38,  -2.31, -10.04>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.57,  -1.84, -10.25>, < -7.38,  -2.31, -10.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.20,  -2.79,  -9.83>, < -7.33,  -3.20, -10.21>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.47,  -3.61, -10.59>, < -7.33,  -3.20, -10.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.20,  -2.79,  -9.83>, < -7.63,  -2.95,  -9.06>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.06,  -3.11,  -8.29>, < -7.63,  -2.95,  -9.06>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.30,   6.74,  -4.44>, <  0.07,   6.28,  -4.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.16,   5.82,  -4.80>, <  0.07,   6.28,  -4.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.30,   6.74,  -4.44>, <  0.84,   6.72,  -4.42>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.39,   6.69,  -4.40>, <  0.84,   6.72,  -4.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.30,   6.74,  -4.44>, <  0.00,   7.43,  -4.94>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.29,   8.13,  -5.44>, <  0.00,   7.43,  -4.94>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.30,   6.74,  -4.44>, < -0.01,   6.81,  -3.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.32,   6.89,  -2.81>, < -0.01,   6.81,  -3.63>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.40,   0.44,  -5.12>, <  7.40,  -0.26,  -5.68>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.39,  -0.97,  -6.23>, <  7.40,  -0.26,  -5.68>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.40,   0.44,  -5.12>, <  7.36,   0.30,  -4.61>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.31,   0.16,  -4.10>, <  7.36,   0.30,  -4.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.40,   0.44,  -5.12>, <  7.87,   0.73,  -5.21>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.34,   1.01,  -5.29>, <  7.87,   0.73,  -5.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.40,   0.44,  -5.12>, <  6.70,   0.98,  -5.37>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.00,   1.52,  -5.63>, <  6.70,   0.98,  -5.37>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.94,  -4.02,  -3.52>, <  5.17,  -4.34,  -3.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.39,  -4.65,  -2.75>, <  5.17,  -4.34,  -3.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.94,  -4.02,  -3.52>, <  4.55,  -4.33,  -3.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.16,  -4.63,  -4.11>, <  4.55,  -4.33,  -3.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.94,  -4.02,  -3.52>, <  5.58,  -3.80,  -4.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.22,  -3.58,  -4.64>, <  5.58,  -3.80,  -4.08>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.94,  -4.02,  -3.52>, <  4.52,  -3.36,  -3.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.10,  -2.70,  -2.63>, <  4.52,  -3.36,  -3.08>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.11,   5.34,  -7.85>, <  2.86,   5.48,  -8.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.61,   5.63,  -8.78>, <  2.86,   5.48,  -8.31>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.11,   5.34,  -7.85>, <  1.86,   4.91,  -8.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.61,   4.48,  -8.29>, <  1.86,   4.91,  -8.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.11,   5.34,  -7.85>, <  2.38,   5.16,  -7.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.66,   4.97,  -6.16>, <  2.38,   5.16,  -7.00>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.11,   5.34,  -7.85>, <  1.77,   5.76,  -7.85>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.43,   6.18,  -7.86>, <  1.77,   5.76,  -7.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,  -5.30, -12.08>, < -1.75,  -4.93, -12.49>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.71,  -4.55, -12.91>, < -1.75,  -4.93, -12.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,  -5.30, -12.08>, < -2.13,  -5.69, -12.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.46,  -6.07, -12.39>, < -2.13,  -5.69, -12.23>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,  -5.30, -12.08>, < -1.01,  -5.64, -11.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.23,  -5.98, -11.58>, < -1.01,  -5.64, -11.83>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,  -5.30, -12.08>, < -2.16,  -4.90, -11.39>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.54,  -4.51, -10.71>, < -2.16,  -4.90, -11.39>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.53,  -2.88,  -5.30>, < -3.51,  -3.14,  -4.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.49,  -3.40,  -4.34>, < -3.51,  -3.14,  -4.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.53,  -2.88,  -5.30>, < -4.09,  -3.33,  -5.81>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.66,  -3.77,  -6.32>, < -4.09,  -3.33,  -5.81>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.53,  -2.88,  -5.30>, < -3.70,  -2.37,  -5.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.88,  -1.86,  -5.16>, < -3.70,  -2.37,  -5.23>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.53,  -2.88,  -5.30>, < -2.73,  -2.88,  -5.70>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.93,  -2.88,  -6.11>, < -2.73,  -2.88,  -5.70>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.70,   3.28,  -8.02>, < -3.23,   3.65,  -8.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.76,   4.03,  -9.30>, < -3.23,   3.65,  -8.66>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.70,   3.28,  -8.02>, < -2.48,   2.83,  -8.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.25,   2.38,  -8.43>, < -2.48,   2.83,  -8.23>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.70,   3.28,  -8.02>, < -2.03,   3.82,  -7.80>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,   4.37,  -7.58>, < -2.03,   3.82,  -7.80>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.70,   3.28,  -8.02>, < -2.98,   3.17,  -7.57>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.25,   3.06,  -7.11>, < -2.98,   3.17,  -7.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.95,   0.13, -13.95>, < -2.77,  -0.36, -14.12>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.59,  -0.85, -14.29>, < -2.77,  -0.36, -14.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.95,   0.13, -13.95>, < -3.42,   0.09, -13.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.88,   0.06, -13.35>, < -3.42,   0.09, -13.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.95,   0.13, -13.95>, < -3.06,   0.61, -14.71>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.16,   1.09, -15.46>, < -3.06,   0.61, -14.71>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.95,   0.13, -13.95>, < -2.32,   0.46, -13.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.69,   0.80, -12.86>, < -2.32,   0.46, -13.40>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.08,   2.26, -16.48>, <  4.18,   1.37, -16.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.27,   0.47, -16.79>, <  4.18,   1.37, -16.63>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.08,   2.26, -16.48>, <  4.86,   2.70, -16.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.63,   3.14, -16.76>, <  4.86,   2.70, -16.62>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.08,   2.26, -16.48>, <  3.70,   2.48, -16.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.31,   2.70, -17.19>, <  3.70,   2.48, -16.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.08,   2.26, -16.48>, <  3.89,   2.33, -15.96>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.70,   2.40, -15.43>, <  3.89,   2.33, -15.96>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.97,   2.94,  -9.53>, <  6.60,   2.34,  -9.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.23,   1.75,  -9.12>, <  6.60,   2.34,  -9.33>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.97,   2.94,  -9.53>, <  5.45,   2.60, -10.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.94,   2.27, -10.80>, <  5.45,   2.60, -10.17>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.97,   2.94,  -9.53>, <  6.18,   3.39,  -9.70>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.39,   3.84,  -9.87>, <  6.18,   3.39,  -9.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.97,   2.94,  -9.53>, <  5.67,   3.03,  -9.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.38,   3.13,  -8.61>, <  5.67,   3.03,  -9.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.59,   4.94,  -1.42>, <  2.73,   4.44,  -0.71>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.88,   3.94,   0.00>, <  2.73,   4.44,  -0.71>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.59,   4.94,  -1.42>, <  2.92,   4.56,  -2.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.26,   4.18,  -2.89>, <  2.92,   4.56,  -2.15>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.59,   4.94,  -1.42>, <  2.85,   5.42,  -1.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.12,   5.89,  -1.24>, <  2.85,   5.42,  -1.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.59,   4.94,  -1.42>, <  2.04,   5.02,  -1.50>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.50,   5.11,  -1.59>, <  2.04,   5.02,  -1.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.26,  -4.38,  -7.14>, <  3.58,  -4.05,  -7.93>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.89,  -3.73,  -8.71>, <  3.58,  -4.05,  -7.93>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.26,  -4.38,  -7.14>, <  3.66,  -4.36,  -6.77>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.06,  -4.33,  -6.40>, <  3.66,  -4.36,  -6.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.26,  -4.38,  -7.14>, <  3.09,  -4.91,  -7.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.92,  -5.43,  -7.32>, <  3.09,  -4.91,  -7.23>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.26,  -4.38,  -7.14>, <  2.58,  -3.90,  -6.84>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.90,  -3.42,  -6.53>, <  2.58,  -3.90,  -6.84>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.67,  -1.88, -12.06>, <  4.49,  -1.36, -12.18>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.30,  -0.85, -12.31>, <  4.49,  -1.36, -12.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.67,  -1.88, -12.06>, <  4.01,  -2.43, -12.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.35,  -2.98, -12.61>, <  4.01,  -2.43, -12.33>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.67,  -1.88, -12.06>, <  5.47,  -2.00, -12.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.26,  -2.12, -12.95>, <  5.47,  -2.00, -12.51>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.67,  -1.88, -12.06>, <  4.79,  -1.94, -11.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.91,  -2.00, -10.97>, <  4.79,  -1.94, -11.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.76,   5.33, -12.17>, <  1.27,   5.15, -12.05>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.78,   4.98, -11.93>, <  1.27,   5.15, -12.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.76,   5.33, -12.17>, <  0.85,   6.01, -12.72>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.94,   6.69, -13.28>, <  0.85,   6.01, -12.72>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.76,   5.33, -12.17>, <  0.35,   5.60, -11.45>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.06,   5.88, -10.72>, <  0.35,   5.60, -11.45>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.76,   5.33, -12.17>, <  0.43,   4.96, -12.45>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.10,   4.60, -12.73>, <  0.43,   4.96, -12.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.12,  -4.69,  -3.58>, < -0.08,  -4.00,  -3.01>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.03,  -3.30,  -2.44>, < -0.08,  -4.00,  -3.01>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.12,  -4.69,  -3.58>, <  0.26,  -5.03,  -3.42>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.65,  -5.37,  -3.26>, <  0.26,  -5.03,  -3.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.12,  -4.69,  -3.58>, < -0.95,  -5.11,  -3.61>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.77,  -5.52,  -3.64>, < -0.95,  -5.11,  -3.61>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.12,  -4.69,  -3.58>, <  0.01,  -4.48,  -4.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.14,  -4.28,  -4.59>, <  0.01,  -4.48,  -4.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.38,   5.67, -12.53>, < -3.11,   4.90, -12.92>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.83,   4.13, -13.30>, < -3.11,   4.90, -12.92>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.38,   5.67, -12.53>, < -3.16,   6.09, -12.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.93,   6.50, -13.13>, < -3.16,   6.09, -12.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.38,   5.67, -12.53>, < -3.24,   5.69, -12.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.09,   5.71, -11.46>, < -3.24,   5.69, -12.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.38,   5.67, -12.53>, < -4.24,   5.79, -12.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.11,   5.92, -12.58>, < -4.24,   5.79, -12.56>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.91,  -1.90, -16.23>, <  1.44,  -1.98, -16.37>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.96,  -2.07, -16.51>, <  1.44,  -1.98, -16.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.91,  -1.90, -16.23>, <  0.88,  -1.28, -15.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.84,  -0.66, -14.92>, <  0.88,  -1.28, -15.58>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.91,  -1.90, -16.23>, <  0.63,  -1.71, -16.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -1.51, -17.08>, <  0.63,  -1.71, -16.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.91,  -1.90, -16.23>, <  0.52,  -2.65, -15.87>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.14,  -3.39, -15.50>, <  0.52,  -2.65, -15.87>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
// no constraints
