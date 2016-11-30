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

set object 15 rect from 0.287055, 0 to 155.949082, 10 fc rgb "#FF0000"
set object 16 rect from 0.043551, 0 to 23.429662, 10 fc rgb "#00FF00"
set object 17 rect from 0.046878, 0 to 24.188763, 10 fc rgb "#0000FF"
set object 18 rect from 0.047979, 0 to 25.118185, 10 fc rgb "#FFFF00"
set object 19 rect from 0.049843, 0 to 25.908630, 10 fc rgb "#FF00FF"
set object 20 rect from 0.051381, 0 to 34.464475, 10 fc rgb "#808080"
set object 21 rect from 0.068330, 0 to 34.763673, 10 fc rgb "#800080"
set object 22 rect from 0.069305, 0 to 35.126042, 10 fc rgb "#008080"
set object 23 rect from 0.069624, 0 to 144.540770, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.290642, 10 to 156.463569, 20 fc rgb "#FF0000"
set object 25 rect from 0.125118, 10 to 64.053401, 20 fc rgb "#00FF00"
set object 26 rect from 0.127075, 10 to 64.481470, 20 fc rgb "#0000FF"
set object 27 rect from 0.127706, 10 to 64.818573, 20 fc rgb "#FFFF00"
set object 28 rect from 0.128372, 10 to 65.337111, 20 fc rgb "#FF00FF"
set object 29 rect from 0.129425, 10 to 73.730219, 20 fc rgb "#808080"
set object 30 rect from 0.145998, 10 to 73.993027, 20 fc rgb "#800080"
set object 31 rect from 0.146754, 10 to 74.250783, 20 fc rgb "#008080"
set object 32 rect from 0.147045, 10 to 146.515358, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.294385, 20 to 164.754089, 30 fc rgb "#FF0000"
set object 34 rect from 0.202866, 20 to 103.129116, 30 fc rgb "#00FF00"
set object 35 rect from 0.204387, 20 to 103.438421, 30 fc rgb "#0000FF"
set object 36 rect from 0.204779, 20 to 104.834825, 30 fc rgb "#FFFF00"
set object 37 rect from 0.207544, 20 to 110.632226, 30 fc rgb "#FF00FF"
set object 38 rect from 0.219021, 20 to 118.927295, 30 fc rgb "#808080"
set object 39 rect from 0.235433, 20 to 119.446843, 30 fc rgb "#800080"
set object 40 rect from 0.236722, 20 to 119.757157, 30 fc rgb "#008080"
set object 41 rect from 0.237068, 20 to 148.530877, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.292922, 30 to 166.787803, 40 fc rgb "#FF0000"
set object 43 rect from 0.155457, 30 to 79.277448, 40 fc rgb "#00FF00"
set object 44 rect from 0.157191, 30 to 79.598373, 40 fc rgb "#0000FF"
set object 45 rect from 0.157603, 30 to 81.161061, 40 fc rgb "#FFFF00"
set object 46 rect from 0.160711, 30 to 89.590061, 40 fc rgb "#FF00FF"
set object 47 rect from 0.177385, 30 to 97.842670, 40 fc rgb "#808080"
set object 48 rect from 0.193733, 30 to 98.378394, 40 fc rgb "#800080"
set object 49 rect from 0.195043, 30 to 98.733179, 40 fc rgb "#008080"
set object 50 rect from 0.195469, 30 to 147.637837, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.295822, 40 to 165.518251, 50 fc rgb "#FF0000"
set object 52 rect from 0.245350, 40 to 124.625128, 50 fc rgb "#00FF00"
set object 53 rect from 0.246914, 40 to 124.885407, 50 fc rgb "#0000FF"
set object 54 rect from 0.247211, 40 to 126.368748, 50 fc rgb "#FFFF00"
set object 55 rect from 0.250176, 40 to 132.122680, 50 fc rgb "#FF00FF"
set object 56 rect from 0.261544, 40 to 140.493052, 50 fc rgb "#808080"
set object 57 rect from 0.278097, 40 to 140.985819, 50 fc rgb "#800080"
set object 58 rect from 0.279348, 40 to 141.304214, 50 fc rgb "#008080"
set object 59 rect from 0.279705, 40 to 149.266225, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.288877, 50 to 163.034734, 60 fc rgb "#FF0000"
set object 61 rect from 0.079371, 50 to 41.061411, 60 fc rgb "#00FF00"
set object 62 rect from 0.081609, 50 to 41.453602, 60 fc rgb "#0000FF"
set object 63 rect from 0.082131, 50 to 43.053689, 60 fc rgb "#FFFF00"
set object 64 rect from 0.085359, 50 to 49.601093, 60 fc rgb "#FF00FF"
set object 65 rect from 0.098289, 50 to 57.972978, 60 fc rgb "#808080"
set object 66 rect from 0.114846, 50 to 58.511730, 60 fc rgb "#800080"
set object 67 rect from 0.116146, 50 to 59.017632, 60 fc rgb "#008080"
set object 68 rect from 0.116915, 50 to 145.636469, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
