#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -18.60*x up 17.77*y
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
atom(< -1.29,  -1.49,  -9.57>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(< -1.79,  -0.71, -11.11>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(< -2.62,  -1.45,  -8.37>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(< -0.44,  -0.93,  -9.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(< -0.96,  -2.56,  -9.73>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  3.28,   0.97, -10.42>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(<  2.26,   1.41,  -9.04>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(<  3.99,   2.38, -11.24>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(<  2.61,   0.46, -11.15>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  4.07,   0.27, -10.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(< -1.38,   7.20,  -5.74>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(<  0.16,   8.06,  -5.43>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(< -1.05,   5.41,  -5.77>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(< -1.79,   7.57,  -6.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -2.13,   7.47,  -4.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  4.01,  -0.30, -15.27>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(<  5.64,   0.32, -15.43>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(<  3.01,   0.29, -16.66>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(<  3.65,   0.16, -14.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(<  4.05,  -1.39, -15.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  3.33,  -7.40,  -8.98>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(<  2.83,  -6.00, -10.05>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(<  2.22,  -7.39,  -7.57>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(<  4.35,  -7.32,  -8.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  3.16,  -8.34,  -9.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(<  4.93,  -4.94,  -4.51>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(<  5.82,  -4.10,  -5.84>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(<  3.81,  -6.22,  -4.96>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(<  4.34,  -4.16,  -3.90>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  5.70,  -5.39,  -3.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  0.61,  -2.11, -13.87>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(<  2.29,  -2.66, -13.51>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(< -0.33,  -3.61, -14.16>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(<  0.59,  -1.43, -14.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(<  0.16,  -1.61, -13.01>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -3.63,   1.79,  -3.24>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(< -2.26,   2.54,  -4.02>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(< -3.13,   0.23,  -2.55>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(< -4.46,   1.71,  -3.98>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -3.94,   2.46,  -2.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(<  3.15,   3.53,  -5.66>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(<  2.13,   4.66,  -4.76>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(<  2.34,   1.93,  -5.65>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(<  4.11,   3.49,  -5.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  3.29,   3.88,  -6.68>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(<  0.43,   2.70, -11.81>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(<  0.87,   1.35, -12.90>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(< -0.85,   3.76, -12.57>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(<  1.34,   3.25, -11.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(<  0.02,   2.25, -10.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -5.50,   2.95,  -8.66>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(< -4.41,   1.69,  -9.34>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(< -7.01,   3.16,  -9.62>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(< -4.90,   3.88,  -8.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -5.84,   2.59,  -7.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(< -8.03,  -4.09, -10.80>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(< -8.31,  -5.63, -11.68>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(< -8.34,  -4.27,  -9.00>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(< -8.73,  -3.31, -11.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -6.98,  -3.70, -10.94>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(< -1.08,  -5.97, -11.47>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(< -2.35,  -4.94, -10.80>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(< -0.67,  -7.23, -10.29>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(< -0.16,  -5.38, -11.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(< -1.41,  -6.39, -12.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(<  7.04,  -2.59, -13.48>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(<  5.56,  -3.26, -12.71>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(<  8.45,  -2.88, -12.48>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(<  6.86,  -1.50, -13.64>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(<  7.21,  -3.12, -14.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(<  1.31,  -3.26,  -4.16>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(<  1.92,  -1.90,  -5.11>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(< -0.26,  -3.89,  -4.81>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(<  1.13,  -2.92,  -3.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(<  2.05,  -4.06,  -4.20>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(< -3.61,  -4.68,  -6.66>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(< -4.50,  -6.26,  -6.41>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(< -4.16,  -3.35,  -5.55>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(< -3.86,  -4.30,  -7.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(< -2.52,  -4.91,  -6.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(< -3.72,  -0.76, -13.86>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(< -4.83,  -0.94, -15.22>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(< -2.27,   0.06, -14.41>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(< -3.48,  -1.78, -13.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(< -4.28,  -0.10, -13.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(<  4.87,   0.12,  -3.84>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(<  4.81,   1.77,  -3.01>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(<  6.01,   0.03,  -5.23>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(<  3.87,  -0.14,  -4.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(<  5.19,  -0.65,  -3.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(< -4.08,   3.38, -14.20>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(< -4.13,   5.11, -14.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(< -4.85,   2.83, -12.67>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(< -3.01,   3.09, -14.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(< -4.67,   2.93, -15.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(< -7.78,  -0.91,  -6.63>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(< -7.05,  -0.14,  -5.15>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(< -6.47,  -0.84,  -7.88>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(< -8.63,  -0.35,  -7.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(< -8.06,  -1.96,  -6.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {< -1.29,  -1.49,  -9.57>, < -1.95,  -1.47,  -8.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.62,  -1.45,  -8.37>, < -1.95,  -1.47,  -8.97>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.29,  -1.49,  -9.57>, < -0.87,  -1.21,  -9.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.44,  -0.93,  -9.19>, < -0.87,  -1.21,  -9.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.29,  -1.49,  -9.57>, < -1.12,  -2.02,  -9.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.96,  -2.56,  -9.73>, < -1.12,  -2.02,  -9.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.29,  -1.49,  -9.57>, < -1.54,  -1.10, -10.34>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,  -0.71, -11.11>, < -1.54,  -1.10, -10.34>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.28,   0.97, -10.42>, <  2.77,   1.19,  -9.73>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.26,   1.41,  -9.04>, <  2.77,   1.19,  -9.73>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.28,   0.97, -10.42>, <  3.68,   0.62, -10.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.07,   0.27, -10.11>, <  3.68,   0.62, -10.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.28,   0.97, -10.42>, <  3.64,   1.68, -10.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.99,   2.38, -11.24>, <  3.64,   1.68, -10.83>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.28,   0.97, -10.42>, <  2.95,   0.71, -10.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.61,   0.46, -11.15>, <  2.95,   0.71, -10.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.38,   7.20,  -5.74>, < -1.22,   6.31,  -5.75>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.05,   5.41,  -5.77>, < -1.22,   6.31,  -5.75>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.38,   7.20,  -5.74>, < -1.59,   7.38,  -6.22>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,   7.57,  -6.70>, < -1.59,   7.38,  -6.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.38,   7.20,  -5.74>, < -1.76,   7.33,  -5.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.13,   7.47,  -4.92>, < -1.76,   7.33,  -5.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.38,   7.20,  -5.74>, < -0.61,   7.63,  -5.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.16,   8.06,  -5.43>, < -0.61,   7.63,  -5.59>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.01,  -0.30, -15.27>, <  3.83,  -0.07, -14.81>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.65,   0.16, -14.35>, <  3.83,  -0.07, -14.81>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.01,  -0.30, -15.27>, <  4.03,  -0.84, -15.25>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.05,  -1.39, -15.23>, <  4.03,  -0.84, -15.25>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.01,  -0.30, -15.27>, <  3.51,  -0.00, -15.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.01,   0.29, -16.66>, <  3.51,  -0.00, -15.97>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.01,  -0.30, -15.27>, <  4.82,   0.01, -15.35>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.64,   0.32, -15.43>, <  4.82,   0.01, -15.35>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.33,  -7.40,  -8.98>, <  2.77,  -7.40,  -8.27>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.22,  -7.39,  -7.57>, <  2.77,  -7.40,  -8.27>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.33,  -7.40,  -8.98>, <  3.84,  -7.36,  -8.84>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.35,  -7.32,  -8.70>, <  3.84,  -7.36,  -8.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.33,  -7.40,  -8.98>, <  3.25,  -7.87,  -9.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.16,  -8.34,  -9.54>, <  3.25,  -7.87,  -9.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.33,  -7.40,  -8.98>, <  3.08,  -6.70,  -9.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.83,  -6.00, -10.05>, <  3.08,  -6.70,  -9.51>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.93,  -4.94,  -4.51>, <  5.32,  -5.17,  -4.20>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.70,  -5.39,  -3.88>, <  5.32,  -5.17,  -4.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.93,  -4.94,  -4.51>, <  4.63,  -4.55,  -4.21>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.34,  -4.16,  -3.90>, <  4.63,  -4.55,  -4.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.93,  -4.94,  -4.51>, <  4.37,  -5.58,  -4.73>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.81,  -6.22,  -4.96>, <  4.37,  -5.58,  -4.73>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.93,  -4.94,  -4.51>, <  5.38,  -4.52,  -5.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.82,  -4.10,  -5.84>, <  5.38,  -4.52,  -5.17>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.61,  -2.11, -13.87>, <  0.14,  -2.86, -14.02>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.33,  -3.61, -14.16>, <  0.14,  -2.86, -14.02>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.61,  -2.11, -13.87>, <  0.38,  -1.86, -13.44>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.16,  -1.61, -13.01>, <  0.38,  -1.86, -13.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.61,  -2.11, -13.87>, <  1.45,  -2.39, -13.69>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,  -2.66, -13.51>, <  1.45,  -2.39, -13.69>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.61,  -2.11, -13.87>, <  0.60,  -1.77, -14.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.59,  -1.43, -14.72>, <  0.60,  -1.77, -14.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.63,   1.79,  -3.24>, < -4.04,   1.75,  -3.61>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.46,   1.71,  -3.98>, < -4.04,   1.75,  -3.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.63,   1.79,  -3.24>, < -3.78,   2.13,  -2.85>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.94,   2.46,  -2.46>, < -3.78,   2.13,  -2.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.63,   1.79,  -3.24>, < -3.38,   1.01,  -2.90>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.13,   0.23,  -2.55>, < -3.38,   1.01,  -2.90>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.63,   1.79,  -3.24>, < -2.94,   2.16,  -3.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.26,   2.54,  -4.02>, < -2.94,   2.16,  -3.63>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.15,   3.53,  -5.66>, <  3.63,   3.51,  -5.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.11,   3.49,  -5.14>, <  3.63,   3.51,  -5.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.15,   3.53,  -5.66>, <  2.64,   4.09,  -5.21>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.13,   4.66,  -4.76>, <  2.64,   4.09,  -5.21>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.15,   3.53,  -5.66>, <  3.22,   3.70,  -6.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.29,   3.88,  -6.68>, <  3.22,   3.70,  -6.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.15,   3.53,  -5.66>, <  2.75,   2.73,  -5.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.34,   1.93,  -5.65>, <  2.75,   2.73,  -5.65>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.43,   2.70, -11.81>, < -0.21,   3.23, -12.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.85,   3.76, -12.57>, < -0.21,   3.23, -12.19>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.43,   2.70, -11.81>, <  0.22,   2.48, -11.36>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.02,   2.25, -10.92>, <  0.22,   2.48, -11.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.43,   2.70, -11.81>, <  0.65,   2.03, -12.36>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.87,   1.35, -12.90>, <  0.65,   2.03, -12.36>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.43,   2.70, -11.81>, <  0.88,   2.98, -11.70>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.34,   3.25, -11.60>, <  0.88,   2.98, -11.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.50,   2.95,  -8.66>, < -5.67,   2.77,  -8.18>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.84,   2.59,  -7.69>, < -5.67,   2.77,  -8.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.50,   2.95,  -8.66>, < -5.20,   3.42,  -8.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.90,   3.88,  -8.60>, < -5.20,   3.42,  -8.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.50,   2.95,  -8.66>, < -6.25,   3.06,  -9.14>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.01,   3.16,  -9.62>, < -6.25,   3.06,  -9.14>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.50,   2.95,  -8.66>, < -4.95,   2.32,  -9.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.41,   1.69,  -9.34>, < -4.95,   2.32,  -9.00>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -8.03,  -4.09, -10.80>, < -8.18,  -4.18,  -9.90>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.34,  -4.27,  -9.00>, < -8.18,  -4.18,  -9.90>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -8.03,  -4.09, -10.80>, < -8.17,  -4.86, -11.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.31,  -5.63, -11.68>, < -8.17,  -4.86, -11.24>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -8.03,  -4.09, -10.80>, < -8.38,  -3.70, -11.02>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.73,  -3.31, -11.23>, < -8.38,  -3.70, -11.02>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -8.03,  -4.09, -10.80>, < -7.50,  -3.89, -10.87>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.98,  -3.70, -10.94>, < -7.50,  -3.89, -10.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.08,  -5.97, -11.47>, < -0.88,  -6.60, -10.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.67,  -7.23, -10.29>, < -0.88,  -6.60, -10.88>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.08,  -5.97, -11.47>, < -1.72,  -5.46, -11.14>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.35,  -4.94, -10.80>, < -1.72,  -5.46, -11.14>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.08,  -5.97, -11.47>, < -1.25,  -6.18, -11.92>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.41,  -6.39, -12.38>, < -1.25,  -6.18, -11.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.08,  -5.97, -11.47>, < -0.62,  -5.67, -11.57>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.16,  -5.38, -11.66>, < -0.62,  -5.67, -11.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.04,  -2.59, -13.48>, <  7.74,  -2.73, -12.98>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.45,  -2.88, -12.48>, <  7.74,  -2.73, -12.98>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.04,  -2.59, -13.48>, <  6.30,  -2.92, -13.10>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.56,  -3.26, -12.71>, <  6.30,  -2.92, -13.10>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.04,  -2.59, -13.48>, <  7.12,  -2.86, -13.95>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.21,  -3.12, -14.43>, <  7.12,  -2.86, -13.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.04,  -2.59, -13.48>, <  6.95,  -2.04, -13.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.86,  -1.50, -13.64>, <  6.95,  -2.04, -13.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -3.26,  -4.16>, <  0.52,  -3.58,  -4.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.26,  -3.89,  -4.81>, <  0.52,  -3.58,  -4.48>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -3.26,  -4.16>, <  1.68,  -3.66,  -4.18>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.05,  -4.06,  -4.20>, <  1.68,  -3.66,  -4.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -3.26,  -4.16>, <  1.22,  -3.09,  -3.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.13,  -2.92,  -3.10>, <  1.22,  -3.09,  -3.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -3.26,  -4.16>, <  1.61,  -2.58,  -4.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.92,  -1.90,  -5.11>, <  1.61,  -2.58,  -4.63>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.61,  -4.68,  -6.66>, < -3.73,  -4.49,  -7.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.86,  -4.30,  -7.69>, < -3.73,  -4.49,  -7.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.61,  -4.68,  -6.66>, < -3.88,  -4.01,  -6.11>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.16,  -3.35,  -5.55>, < -3.88,  -4.01,  -6.11>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.61,  -4.68,  -6.66>, < -4.05,  -5.47,  -6.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.50,  -6.26,  -6.41>, < -4.05,  -5.47,  -6.54>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.61,  -4.68,  -6.66>, < -3.06,  -4.79,  -6.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.52,  -4.91,  -6.53>, < -3.06,  -4.79,  -6.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.72,  -0.76, -13.86>, < -3.60,  -1.27, -13.69>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.48,  -1.78, -13.51>, < -3.60,  -1.27, -13.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.72,  -0.76, -13.86>, < -4.27,  -0.85, -14.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.83,  -0.94, -15.22>, < -4.27,  -0.85, -14.54>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.72,  -0.76, -13.86>, < -3.00,  -0.35, -14.14>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.27,   0.06, -14.41>, < -3.00,  -0.35, -14.14>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.72,  -0.76, -13.86>, < -4.00,  -0.43, -13.50>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.28,  -0.10, -13.13>, < -4.00,  -0.43, -13.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.87,   0.12,  -3.84>, <  5.44,   0.08,  -4.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.01,   0.03,  -5.23>, <  5.44,   0.08,  -4.54>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.87,   0.12,  -3.84>, <  5.03,  -0.26,  -3.50>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.19,  -0.65,  -3.16>, <  5.03,  -0.26,  -3.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.87,   0.12,  -3.84>, <  4.84,   0.95,  -3.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.81,   1.77,  -3.01>, <  4.84,   0.95,  -3.43>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.87,   0.12,  -3.84>, <  4.37,  -0.01,  -4.05>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.87,  -0.14,  -4.26>, <  4.37,  -0.01,  -4.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.08,   3.38, -14.20>, < -4.46,   3.11, -13.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.85,   2.83, -12.67>, < -4.46,   3.11, -13.43>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.08,   3.38, -14.20>, < -4.38,   3.16, -14.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.67,   2.93, -15.04>, < -4.38,   3.16, -14.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.08,   3.38, -14.20>, < -3.55,   3.24, -14.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.01,   3.09, -14.25>, < -3.55,   3.24, -14.23>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.08,   3.38, -14.20>, < -4.11,   4.24, -14.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.13,   5.11, -14.45>, < -4.11,   4.24, -14.33>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.78,  -0.91,  -6.63>, < -7.92,  -1.44,  -6.55>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.06,  -1.96,  -6.47>, < -7.92,  -1.44,  -6.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.78,  -0.91,  -6.63>, < -7.13,  -0.88,  -7.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.47,  -0.84,  -7.88>, < -7.13,  -0.88,  -7.26>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.78,  -0.91,  -6.63>, < -8.21,  -0.63,  -6.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.63,  -0.35,  -7.03>, < -8.21,  -0.63,  -6.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.78,  -0.91,  -6.63>, < -7.42,  -0.53,  -5.89>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.05,  -0.14,  -5.15>, < -7.42,  -0.53,  -5.89>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
// no constraints
