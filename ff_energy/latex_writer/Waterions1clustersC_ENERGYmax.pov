#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -15.96*x up 7.55*y
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
atom(<  0.70,  -1.83,  -0.38>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  0.82,  -1.04,  -0.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -0.19,  -1.79,  -0.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(<  4.00,   2.47,  -3.13>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(<  3.38,   2.89,  -2.55>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  3.52,   2.59,  -3.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  6.49,   3.33,  -3.98>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  5.74,   3.10,  -3.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  7.31,   3.09,  -3.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  1.41,   0.60,  -2.13>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  0.72,   0.94,  -2.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  1.53,   1.36,  -1.56>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  1.71,  -1.79,  -4.89>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  1.74,  -2.49,  -5.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  1.13,  -2.29,  -4.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  5.81,  -0.36,  -1.92>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  6.76,  -0.63,  -1.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(<  5.76,   0.47,  -1.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  7.33,   0.36,  -4.22>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  6.66,  -0.08,  -4.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  7.05,   0.31,  -3.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(<  3.79,  -2.56,  -2.64>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(<  4.47,  -2.52,  -1.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(<  3.52,  -3.47,  -2.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  5.19,  -1.19,  -5.51>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  5.49,  -1.24,  -6.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  4.54,  -1.96,  -5.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  0.92,  -0.37,  -7.01>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  1.16,   0.57,  -6.93>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  1.31,  -0.63,  -6.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  5.07,  -2.58,  -0.40>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(<  5.48,  -1.70,  -0.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  4.22,  -2.32,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  5.47,   1.78,  -6.16>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  5.81,   2.37,  -5.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  4.49,   1.91,  -6.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  3.55,  -0.12,  -3.62>, 0.81, rgb <0.50, 0.20, 0.39>, 0.0, jmol) // #36
atom(<  2.13,   2.18,  -5.46>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
cylinder {<  0.70,  -1.83,  -0.38>, <  0.76,  -1.44,  -0.67>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.82,  -1.04,  -0.96>, <  0.76,  -1.44,  -0.67>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.70,  -1.83,  -0.38>, <  0.26,  -1.81,  -0.21>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.19,  -1.79,  -0.04>, <  0.26,  -1.81,  -0.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.00,   2.47,  -3.13>, <  3.69,   2.68,  -2.84>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.38,   2.89,  -2.55>, <  3.69,   2.68,  -2.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.00,   2.47,  -3.13>, <  3.78,   1.18,  -3.38>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  3.78,   1.18,  -3.38>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  4.00,   2.47,  -3.13>, <  3.76,   2.53,  -3.54>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.52,   2.59,  -3.96>, <  3.76,   2.53,  -3.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.52,   2.59,  -3.96>, <  3.53,   1.24,  -3.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  3.53,   1.24,  -3.79>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  6.49,   3.33,  -3.98>, <  6.12,   3.22,  -3.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.74,   3.10,  -3.38>, <  6.12,   3.22,  -3.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.49,   3.33,  -3.98>, <  6.90,   3.21,  -3.75>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  7.31,   3.09,  -3.52>, <  6.90,   3.21,  -3.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.41,   0.60,  -2.13>, <  1.47,   0.98,  -1.84>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.53,   1.36,  -1.56>, <  1.47,   0.98,  -1.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.41,   0.60,  -2.13>, <  1.07,   0.77,  -2.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.72,   0.94,  -2.81>, <  1.07,   0.77,  -2.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.41,   0.60,  -2.13>, <  2.48,   0.24,  -2.88>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  2.48,   0.24,  -2.88>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.72,   0.94,  -2.81>, <  2.14,   0.41,  -3.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  2.14,   0.41,  -3.22>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.71,  -1.79,  -4.89>, <  2.63,  -0.95,  -4.26>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  2.63,  -0.95,  -4.26>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.71,  -1.79,  -4.89>, <  1.42,  -2.04,  -4.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.13,  -2.29,  -4.22>, <  1.42,  -2.04,  -4.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.71,  -1.79,  -4.89>, <  1.72,  -2.14,  -5.24>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.74,  -2.49,  -5.58>, <  1.72,  -2.14,  -5.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.81,  -0.36,  -1.92>, <  4.68,  -0.24,  -2.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  4.68,  -0.24,  -2.77>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  5.81,  -0.36,  -1.92>, <  6.29,  -0.50,  -1.83>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  6.76,  -0.63,  -1.74>, <  6.29,  -0.50,  -1.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.81,  -0.36,  -1.92>, <  5.79,   0.05,  -1.69>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.76,   0.47,  -1.46>, <  5.79,   0.05,  -1.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.76,   0.47,  -1.46>, <  4.65,   0.18,  -2.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  4.65,   0.18,  -2.54>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  7.33,   0.36,  -4.22>, <  7.19,   0.34,  -3.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  7.05,   0.31,  -3.32>, <  7.19,   0.34,  -3.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.33,   0.36,  -4.22>, <  7.00,   0.14,  -4.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  6.66,  -0.08,  -4.71>, <  7.00,   0.14,  -4.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.79,  -2.56,  -2.64>, <  3.66,  -3.02,  -2.58>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.52,  -3.47,  -2.51>, <  3.66,  -3.02,  -2.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.79,  -2.56,  -2.64>, <  3.67,  -1.34,  -3.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  3.67,  -1.34,  -3.13>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  3.79,  -2.56,  -2.64>, <  4.13,  -2.54,  -2.25>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.47,  -2.52,  -1.87>, <  4.13,  -2.54,  -2.25>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.47,  -2.52,  -1.87>, <  4.01,  -1.32,  -2.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  4.01,  -1.32,  -2.75>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  4.47,  -2.52,  -1.87>, <  4.77,  -2.55,  -1.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.07,  -2.58,  -0.40>, <  4.77,  -2.55,  -1.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.19,  -1.19,  -5.51>, <  4.37,  -0.66,  -4.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  4.37,  -0.66,  -4.57>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  5.19,  -1.19,  -5.51>, <  4.86,  -1.58,  -5.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.54,  -1.96,  -5.58>, <  4.86,  -1.58,  -5.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.19,  -1.19,  -5.51>, <  5.34,  -1.22,  -5.96>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.49,  -1.24,  -6.41>, <  5.34,  -1.22,  -5.96>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.54,  -1.96,  -5.58>, <  4.04,  -1.04,  -4.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  4.04,  -1.04,  -4.60>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.92,  -0.37,  -7.01>, <  1.04,   0.10,  -6.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.16,   0.57,  -6.93>, <  1.04,   0.10,  -6.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.92,  -0.37,  -7.01>, <  1.12,  -0.50,  -6.58>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -0.63,  -6.16>, <  1.12,  -0.50,  -6.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.07,  -2.58,  -0.40>, <  4.65,  -2.45,  -0.20>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.22,  -2.32,   0.00>, <  4.65,  -2.45,  -0.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.07,  -2.58,  -0.40>, <  5.28,  -2.14,  -0.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.48,  -1.70,  -0.61>, <  5.28,  -2.14,  -0.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.47,   1.78,  -6.16>, <  4.98,   1.85,  -6.14>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.49,   1.91,  -6.12>, <  4.98,   1.85,  -6.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.47,   1.78,  -6.16>, <  5.64,   2.07,  -5.82>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.81,   2.37,  -5.47>, <  5.64,   2.07,  -5.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,  -0.12,  -3.62>, <  2.84,   1.03,  -4.54>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  2.13,   2.18,  -5.46>, <  2.84,   1.03,  -4.54>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
// no constraints
