#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -10.56*x up 7.45*y
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
atom(<  3.35,  -0.13,  -3.43>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  3.03,  -0.11,  -4.33>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(<  3.74,   0.77,  -3.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -1.79,   3.28,  -7.02>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -2.28,   2.64,  -6.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -0.83,   3.04,  -6.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(< -0.74,  -0.99,  -2.95>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(< -0.68,  -0.90,  -3.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  0.13,  -1.30,  -2.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -4.15,  -0.99,  -5.82>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -4.90,  -0.55,  -5.45>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -3.56,  -0.18,  -5.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  4.68,  -0.17,  -6.34>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  4.90,   0.07,  -7.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  4.11,  -0.97,  -6.48>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -2.32,   1.09,  -5.61>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -2.58,   1.30,  -4.68>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -1.49,   0.74,  -5.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  2.99,  -2.47,  -6.09>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  2.13,  -2.12,  -5.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  2.74,  -3.39,  -6.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(<  1.96,   0.70,  -8.29>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(<  1.89,   1.39,  -7.67>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(<  1.62,  -0.02,  -7.64>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  0.50,   2.81,  -5.41>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  0.59,   1.86,  -5.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  1.50,   2.93,  -5.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(< -2.79,  -3.28,  -6.47>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(< -3.21,  -2.60,  -5.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -1.95,  -2.85,  -6.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -0.39,  -2.79,  -7.70>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -0.21,  -2.06,  -7.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  0.25,  -2.64,  -8.42>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  1.76,  -2.03,  -2.46>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  2.31,  -2.76,  -2.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  2.36,  -1.30,  -2.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  0.34,  -0.46,  -5.74>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
cylinder {<  3.35,  -0.13,  -3.43>, <  3.19,  -0.12,  -3.88>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.03,  -0.11,  -4.33>, <  3.19,  -0.12,  -3.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.35,  -0.13,  -3.43>, <  3.54,   0.32,  -3.39>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.74,   0.77,  -3.36>, <  3.54,   0.32,  -3.39>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,   3.28,  -7.02>, < -1.31,   3.16,  -6.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.83,   3.04,  -6.70>, < -1.31,   3.16,  -6.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,   3.28,  -7.02>, < -2.03,   2.96,  -6.71>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.28,   2.64,  -6.41>, < -2.03,   2.96,  -6.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.74,  -0.99,  -2.95>, < -0.31,  -1.14,  -2.83>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.13,  -1.30,  -2.70>, < -0.31,  -1.14,  -2.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.74,  -0.99,  -2.95>, < -0.71,  -0.94,  -3.44>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.68,  -0.90,  -3.92>, < -0.71,  -0.94,  -3.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.15,  -0.99,  -5.82>, < -3.86,  -0.59,  -5.89>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.56,  -0.18,  -5.96>, < -3.86,  -0.59,  -5.89>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.15,  -0.99,  -5.82>, < -4.53,  -0.77,  -5.63>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.90,  -0.55,  -5.45>, < -4.53,  -0.77,  -5.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.68,  -0.17,  -6.34>, <  4.40,  -0.57,  -6.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.11,  -0.97,  -6.48>, <  4.40,  -0.57,  -6.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.68,  -0.17,  -6.34>, <  4.79,  -0.05,  -6.80>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.90,   0.07,  -7.26>, <  4.79,  -0.05,  -6.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.32,   1.09,  -5.61>, < -2.45,   1.19,  -5.15>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.58,   1.30,  -4.68>, < -2.45,   1.19,  -5.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.32,   1.09,  -5.61>, < -1.90,   0.92,  -5.49>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.49,   0.74,  -5.36>, < -1.90,   0.92,  -5.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.99,  -2.47,  -6.09>, <  2.87,  -2.93,  -6.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.74,  -3.39,  -6.06>, <  2.87,  -2.93,  -6.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.99,  -2.47,  -6.09>, <  2.56,  -2.29,  -5.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.13,  -2.12,  -5.85>, <  2.56,  -2.29,  -5.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.96,   0.70,  -8.29>, <  1.79,   0.34,  -7.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.62,  -0.02,  -7.64>, <  1.79,   0.34,  -7.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.96,   0.70,  -8.29>, <  1.92,   1.04,  -7.98>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.89,   1.39,  -7.67>, <  1.92,   1.04,  -7.98>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.50,   2.81,  -5.41>, <  0.54,   2.33,  -5.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.59,   1.86,  -5.17>, <  0.54,   2.33,  -5.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.50,   2.81,  -5.41>, <  1.00,   2.87,  -5.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.50,   2.93,  -5.60>, <  1.00,   2.87,  -5.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.79,  -3.28,  -6.47>, < -2.37,  -3.07,  -6.63>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.95,  -2.85,  -6.79>, < -2.37,  -3.07,  -6.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.79,  -3.28,  -6.47>, < -3.00,  -2.94,  -6.17>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.21,  -2.60,  -5.88>, < -3.00,  -2.94,  -6.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.39,  -2.79,  -7.70>, < -0.30,  -2.42,  -7.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.21,  -2.06,  -7.04>, < -0.30,  -2.42,  -7.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.39,  -2.79,  -7.70>, < -0.07,  -2.71,  -8.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.25,  -2.64,  -8.42>, < -0.07,  -2.71,  -8.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.76,  -2.03,  -2.46>, <  2.04,  -2.39,  -2.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.31,  -2.76,  -2.65>, <  2.04,  -2.39,  -2.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.76,  -2.03,  -2.46>, <  2.06,  -1.67,  -2.59>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.36,  -1.30,  -2.72>, <  2.06,  -1.67,  -2.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
