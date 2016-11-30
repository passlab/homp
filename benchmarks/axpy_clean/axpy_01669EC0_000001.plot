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

set object 15 rect from 0.283007, 0 to 156.521894, 10 fc rgb "#FF0000"
set object 16 rect from 0.037479, 0 to 20.788220, 10 fc rgb "#00FF00"
set object 17 rect from 0.041001, 0 to 21.528274, 10 fc rgb "#0000FF"
set object 18 rect from 0.042067, 0 to 22.508513, 10 fc rgb "#FFFF00"
set object 19 rect from 0.044007, 0 to 23.261399, 10 fc rgb "#FF00FF"
set object 20 rect from 0.045441, 0 to 32.206714, 10 fc rgb "#808080"
set object 21 rect from 0.062885, 0 to 32.522853, 10 fc rgb "#800080"
set object 22 rect from 0.063918, 0 to 32.960625, 10 fc rgb "#008080"
set object 23 rect from 0.064348, 0 to 144.674882, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.291016, 10 to 159.042281, 20 fc rgb "#FF0000"
set object 25 rect from 0.254314, 10 to 131.295920, 20 fc rgb "#00FF00"
set object 26 rect from 0.256180, 10 to 131.628478, 20 fc rgb "#0000FF"
set object 27 rect from 0.256589, 10 to 131.952830, 20 fc rgb "#FFFF00"
set object 28 rect from 0.257255, 10 to 132.496321, 20 fc rgb "#FF00FF"
set object 29 rect from 0.258301, 10 to 141.101889, 20 fc rgb "#808080"
set object 30 rect from 0.275065, 10 to 141.371325, 20 fc rgb "#800080"
set object 31 rect from 0.275866, 10 to 141.661287, 20 fc rgb "#008080"
set object 32 rect from 0.276140, 10 to 149.094673, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.284767, 20 to 163.569855, 30 fc rgb "#FF0000"
set object 34 rect from 0.074016, 20 to 38.969836, 30 fc rgb "#00FF00"
set object 35 rect from 0.076287, 20 to 39.383488, 30 fc rgb "#0000FF"
set object 36 rect from 0.076853, 20 to 41.123794, 30 fc rgb "#FFFF00"
set object 37 rect from 0.080282, 20 to 47.784792, 30 fc rgb "#FF00FF"
set object 38 rect from 0.093231, 20 to 56.220479, 30 fc rgb "#808080"
set object 39 rect from 0.109667, 20 to 56.780397, 30 fc rgb "#800080"
set object 40 rect from 0.111042, 20 to 57.349040, 30 fc rgb "#008080"
set object 41 rect from 0.111887, 20 to 145.819342, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.286580, 30 to 164.213933, 40 fc rgb "#FF0000"
set object 43 rect from 0.120361, 30 to 62.591516, 40 fc rgb "#00FF00"
set object 44 rect from 0.122317, 30 to 62.914323, 40 fc rgb "#0000FF"
set object 45 rect from 0.122710, 30 to 64.478599, 40 fc rgb "#FFFF00"
set object 46 rect from 0.125779, 30 to 71.213503, 40 fc rgb "#FF00FF"
set object 47 rect from 0.138880, 30 to 79.597868, 40 fc rgb "#808080"
set object 48 rect from 0.155207, 30 to 80.109545, 40 fc rgb "#800080"
set object 49 rect from 0.156465, 30 to 80.437484, 40 fc rgb "#008080"
set object 50 rect from 0.156859, 30 to 146.682054, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.288295, 40 to 167.190056, 50 fc rgb "#FF0000"
set object 52 rect from 0.165643, 40 to 85.770804, 50 fc rgb "#00FF00"
set object 53 rect from 0.167484, 40 to 86.093099, 50 fc rgb "#0000FF"
set object 54 rect from 0.167861, 40 to 90.329679, 50 fc rgb "#FFFF00"
set object 55 rect from 0.176176, 40 to 96.447179, 50 fc rgb "#FF00FF"
set object 56 rect from 0.188058, 40 to 104.892624, 50 fc rgb "#808080"
set object 57 rect from 0.204505, 40 to 105.404293, 50 fc rgb "#800080"
set object 58 rect from 0.205740, 40 to 105.725052, 50 fc rgb "#008080"
set object 59 rect from 0.206118, 40 to 147.657161, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.289624, 50 to 164.904730, 60 fc rgb "#FF0000"
set object 61 rect from 0.212490, 50 to 109.718373, 60 fc rgb "#00FF00"
set object 62 rect from 0.214127, 50 to 110.065301, 60 fc rgb "#0000FF"
set object 63 rect from 0.214573, 50 to 111.432501, 60 fc rgb "#FFFF00"
set object 64 rect from 0.217248, 50 to 117.474054, 60 fc rgb "#FF00FF"
set object 65 rect from 0.229019, 50 to 125.843536, 60 fc rgb "#808080"
set object 66 rect from 0.245328, 50 to 126.346488, 60 fc rgb "#800080"
set object 67 rect from 0.246551, 50 to 126.658520, 60 fc rgb "#008080"
set object 68 rect from 0.246907, 50 to 148.418261, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
