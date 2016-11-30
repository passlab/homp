set title "Offloading (axpy_kernel) Profile on 6 Devices"
set yrange [0:72.000000]
set xlabel "execution time in ms"
set xrange [0:158.400000]
set style fill pattern 2 bo 1
set style rect fs solid 1 noborder
set border 15 lw 0.2
set xtics out nomirror
unset key
set ytics out nomirror ("dev 0(sysid:0,type:HOSTCPU)" 5,"dev 1(sysid:1,type:HOSTCPU)" 15,"dev 2(sysid:0,type:THSIM)" 25,"dev 3(sysid:1,type:THSIM)" 35,"dev 4(sysid:2,type:THSIM)" 45,"dev 5(sysid:3,type:THSIM)" 55)
set object 1 rect from 4, 65 to 17, 68 fc rgb "#FF0000"
set label "ACCU_TOTAL" at 4,63 font "Helvetica,8'"

set object 2 rect from 21, 65 to 34, 68 fc rgb "#00FF00"
set label "INIT_0" at 21,63 font "Helvetica,8'"

set object 3 rect from 38, 65 to 51, 68 fc rgb "#0000FF"
set label "INIT_0.1" at 38,63 font "Helvetica,8'"

set object 4 rect from 55, 65 to 68, 68 fc rgb "#FFFF00"
set label "INIT_1" at 55,63 font "Helvetica,8'"

set object 5 rect from 72, 65 to 85, 68 fc rgb "#00FFFF"
set label "MODELING" at 72,63 font "Helvetica,8'"

set object 6 rect from 89, 65 to 102, 68 fc rgb "#FF00FF"
set label "ACC_MAPTO" at 89,63 font "Helvetica,8'"

set object 7 rect from 106, 65 to 119, 68 fc rgb "#808080"
set label "KERN" at 106,63 font "Helvetica,8'"

set object 8 rect from 123, 65 to 136, 68 fc rgb "#800000"
set label "PRE_BAR_X" at 123,63 font "Helvetica,8'"

set object 9 rect from 140, 65 to 153, 68 fc rgb "#808000"
set label "DATA_X" at 140,63 font "Helvetica,8'"

set object 10 rect from 157, 65 to 170, 68 fc rgb "#008000"
set label "POST_BAR_X" at 157,63 font "Helvetica,8'"

set object 11 rect from 174, 65 to 187, 68 fc rgb "#800080"
set label "ACC_MAPFROM" at 174,63 font "Helvetica,8'"

set object 12 rect from 191, 65 to 204, 68 fc rgb "#008080"
set label "FINI_1" at 191,63 font "Helvetica,8'"

set object 13 rect from 208, 65 to 221, 68 fc rgb "#000080"
set label "BAR_FINI_2" at 208,63 font "Helvetica,8'"

set object 14 rect from 225, 65 to 238, 68 fc rgb "(null)"
set label "PROF_BAR" at 225,63 font "Helvetica,8'"

set object 15 rect from 0.348777, 0 to 159.519356, 10 fc rgb "#FF0000"
set object 16 rect from 0.302899, 0 to 130.373894, 10 fc rgb "#00FF00"
set object 17 rect from 0.305212, 0 to 130.654439, 10 fc rgb "#0000FF"
set object 18 rect from 0.305623, 0 to 131.043182, 10 fc rgb "#FFFF00"
set object 19 rect from 0.306548, 0 to 131.487093, 10 fc rgb "#FF00FF"
set object 20 rect from 0.307593, 0 to 140.801957, 10 fc rgb "#808080"
set object 21 rect from 0.329387, 0 to 141.053422, 10 fc rgb "#800080"
set object 22 rect from 0.330196, 0 to 141.301037, 10 fc rgb "#008080"
set object 23 rect from 0.330524, 0 to 148.906544, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.341842, 10 to 156.876846, 20 fc rgb "#FF0000"
set object 25 rect from 0.104465, 10 to 45.722425, 20 fc rgb "#00FF00"
set object 26 rect from 0.107272, 10 to 46.071824, 20 fc rgb "#0000FF"
set object 27 rect from 0.107842, 10 to 46.483660, 20 fc rgb "#FFFF00"
set object 28 rect from 0.108852, 10 to 47.059718, 20 fc rgb "#FF00FF"
set object 29 rect from 0.110196, 10 to 56.497322, 20 fc rgb "#808080"
set object 30 rect from 0.132342, 10 to 56.794973, 20 fc rgb "#800080"
set object 31 rect from 0.133210, 10 to 57.057556, 20 fc rgb "#008080"
set object 32 rect from 0.133549, 10 to 145.754262, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.345483, 20 to 165.893632, 30 fc rgb "#FF0000"
set object 34 rect from 0.196270, 20 to 84.725026, 30 fc rgb "#00FF00"
set object 35 rect from 0.198455, 20 to 84.976918, 30 fc rgb "#0000FF"
set object 36 rect from 0.198833, 20 to 86.356978, 30 fc rgb "#FFFF00"
set object 37 rect from 0.202045, 20 to 93.431331, 30 fc rgb "#FF00FF"
set object 38 rect from 0.218587, 20 to 102.801792, 30 fc rgb "#808080"
set object 39 rect from 0.240522, 20 to 103.181554, 30 fc rgb "#800080"
set object 40 rect from 0.241670, 20 to 103.508713, 30 fc rgb "#008080"
set object 41 rect from 0.242169, 20 to 147.400754, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.347166, 30 to 164.709442, 40 fc rgb "#FF0000"
set object 43 rect from 0.250557, 30 to 107.855023, 40 fc rgb "#00FF00"
set object 44 rect from 0.252548, 30 to 108.119745, 40 fc rgb "#0000FF"
set object 45 rect from 0.252926, 30 to 109.437794, 40 fc rgb "#FFFF00"
set object 46 rect from 0.256027, 30 to 114.787394, 40 fc rgb "#FF00FF"
set object 47 rect from 0.268522, 30 to 124.036826, 40 fc rgb "#808080"
set object 48 rect from 0.290168, 30 to 124.402904, 40 fc rgb "#800080"
set object 49 rect from 0.291293, 30 to 124.701410, 40 fc rgb "#008080"
set object 50 rect from 0.291726, 30 to 148.224854, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.343603, 40 to 163.453402, 50 fc rgb "#FF0000"
set object 52 rect from 0.142692, 40 to 61.960680, 50 fc rgb "#00FF00"
set object 53 rect from 0.145256, 40 to 62.263035, 50 fc rgb "#0000FF"
set object 54 rect from 0.145719, 40 to 63.686288, 50 fc rgb "#FFFF00"
set object 55 rect from 0.149058, 40 to 68.955488, 50 fc rgb "#FF00FF"
set object 56 rect from 0.161366, 40 to 78.389669, 50 fc rgb "#808080"
set object 57 rect from 0.183470, 40 to 78.823744, 50 fc rgb "#800080"
set object 58 rect from 0.184730, 40 to 79.198803, 50 fc rgb "#008080"
set object 59 rect from 0.185327, 40 to 146.673303, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.339550, 50 to 163.154897, 60 fc rgb "#FF0000"
set object 61 rect from 0.043453, 50 to 20.384847, 60 fc rgb "#00FF00"
set object 62 rect from 0.048269, 50 to 21.116573, 60 fc rgb "#0000FF"
set object 63 rect from 0.049508, 50 to 23.180888, 60 fc rgb "#FFFF00"
set object 64 rect from 0.054370, 50 to 28.985091, 60 fc rgb "#FF00FF"
set object 65 rect from 0.067910, 50 to 38.314497, 60 fc rgb "#808080"
set object 66 rect from 0.089736, 50 to 38.776798, 60 fc rgb "#800080"
set object 67 rect from 0.091402, 50 to 39.343020, 60 fc rgb "#008080"
set object 68 rect from 0.092139, 50 to 144.584612, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
