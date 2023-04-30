import random
import numpy as np
import CoolProp.CoolProp as coolprop

best_fit = [12.322349504376072, 0.9121870989330239]


def point2(T4, RH4):
    Q4 = 8.333333333
    Qsr = 100  # kW
    Qlr = 30  # kW
    Qvent = 70  # kW
    T1 = 33.2 + 273.15
    T2_goal = 25 + 273.15
    Qvent_s = 1.1 * 6000 * (T1 - T2_goal) * 9/5  # Btu/Hr
    Qvent_s = Qvent_s * 2.931 * 10 ** -4  # kW
    Qvent_l = Qvent - Qvent_s  # kW
    Qst = Qsr + Qvent_s  # kW
    Qlt = Qlr + Qvent_l  # kW

    v4 = coolprop.HAPropsSI('V', 'T', T4, 'P', 101325, 'R', RH4)
    m4_dot = Q4/v4

    T2 = (Qst/m4_dot) + T4

    w4 = coolprop.HAPropsSI('W', 'T', T4, 'P', 101325, 'R', RH4)
    hlr = coolprop.HAPropsSI('H', 'T', T2, 'P', 101325, 'W', w4)/1000
    h2 = (Qlt/m4_dot) + hlr
    RH2 = coolprop.HAPropsSI('R', 'T', T2, 'P', 101325, 'H', h2*1000)

    return T2, RH2


T2, RH2 = point2(15 + 273.15, 0.90)
print(T2-273.15, RH2)


T2, RH2 = point2(best_fit[0] + 273.15,  best_fit[1])
print(T2-273.15, RH2)
