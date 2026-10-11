"""`frc_coil_gui.m` 的移植（Fig4/gui 与 Fig5/gui 两份**逐字节相同**，合并后的实现）。

源文件：

    gseq-main/GSEQ-FRC-open/second_paper/Fig4/gui/frc_coil_gui.m
    gseq-main/GSEQ-FRC-open/second_paper/Fig5/gui/frc_coil_gui.m   （MD5 相同）

它是"FRC 线圈设计与磁场"的界面：一个图窗 + 3 个坐标区 + 36 个控件
（输入框、输出框、method 下拉、Load/Save/Run/Save Fig/Exit 按钮）。

**回调是字符串**（MATLAB 的老写法：`uicontrol(..., 'Callback', run)`，`run` 是一段
拼起来的代码）。移植版把这段字符串**原样保留**（`str2num`/`get`/`set`/`fun_frc_coil`
等等都在里面），但不执行它——这和 `mplot.uicontrol` 的口径一致：交互不参与验收
（§3），控件属性照抄下来供以后接真控件时用。

于是它跑起来的效果是：**把界面搭出来**（控件数、文字、位置都能对），并把
`hAxes1/hAxes2/hAxes3` 与各句柄留在工作区里——同目录的 `plt_wall*.m`、
`fun_frc_coil.m` 正是接着用这些名字（会话链：GUI → Run → plt_wall）。

验收：§7「绘图/GUI 脚本」那一节的三条（能跑通 / 结构对 / 人眼比对论文图），
其中"结构"这里落成"控件数、各控件的 String 与 Position 跟源码一致"。
"""

from __future__ import annotations

from gseqfrc import mplot


def run(ws) -> None:
    mplot.close_all()  # close all; clear;clc;

    # 回调字符串：原文是 `run=['...'...]` 拼起来的，这里逐段照抄（`''` 是 MATLAB
    # 单引号字符串里的转义，写成一个 `'`）。
    run_callback = (
        "r=str2num(get(heditr,'String'));"
        "z=str2num(get(heditz,'String'));"
        "rw=str2num(get(heditrw,'String'));"
        "zw=str2num(get(heditzw,'String'));"
        "rl=str2num(get(heditrl,'String'));"
        "zl=str2num(get(heditzl,'String'));"
        "hl=str2num(get(hedithl,'String'));"
        "wl=str2num(get(heditwl,'String'));"
        "Il=str2num(get(heditIl,'String'));"
        "Nl=str2num(get(heditNl,'String'));save('frc_coil.mat','rl','zl','hl','wl','Il','Nl');"
        "imethod=get(hpopimethod,'value');"
        "[runtime,psi0,Br0,Bz0]=fun_frc_coil(r,z,rw,zw,rl,zl,hl,wl,Il,Nl,imethod,hAxes1,hAxes2,hAxes3);"
        "set(heditBr,'String',num2str(Br0));"
        "set(heditBz,'String',num2str(Bz0));"
        "set(heditpsi,'String',num2str(psi0));"
        "set(heditB,'String',num2str(sqrt(Br0^2+Bz0^2)));"
    )
    savefig_callback = (
        "set(gcf,'PaperPositionMode','auto');"
        "print(gcf,'-dpng',['frc_Bfield_imethod=',num2str(imethod),'.png']);"
    )
    loaddat_callback = (
        "[filename,~]=uigetfile('*.dat');fid = fopen(filename);"
        "set(heditr,'String',fgetl(fid));"
        "set(heditz,'String',fgetl(fid));"
        "set(heditrw,'String',fgetl(fid));"
        "set(heditzw,'String',fgetl(fid));"
        "set(heditrl,'String',fgetl(fid));"
        "set(heditzl,'String',fgetl(fid));"
        "set(hedithl,'String',fgetl(fid));"
        "set(heditwl,'String',fgetl(fid));"
        "set(heditIl,'String',fgetl(fid));"
        "set(heditNl,'String',fgetl(fid));"
        "fclose(fid);"
    )
    savedat_callback = (
        "[filename,pathname] = uiputfile('*.dat');"
        "fid=fopen(filename,'w');"
        "fprintf(fid,'%s\\n',get(heditr,'String'));"
        "fprintf(fid,'%s\\n',get(heditz,'String'));"
        "fprintf(fid,'%s\\n',get(heditrw,'String'));"
        "fprintf(fid,'%s\\n',get(heditzw,'String'));"
        "fprintf(fid,'%s\\n',get(heditrl,'String'));"
        "fprintf(fid,'%s\\n',get(heditzl,'String'));"
        "fprintf(fid,'%s\\n',get(hedithl,'String'));"
        "fprintf(fid,'%s\\n',get(heditwl,'String'));"
        "fprintf(fid,'%s\\n',get(heditIl,'String'));"
        "fprintf(fid,'%s\\n',get(heditNl,'String'));"
        "fclose(fid);"
    )

    hf = mplot.figure("unit", "normalized", "Position", [-0.03, 0.1, 0.5, 0.6],
                      "Name", "FRC Coils Design and Magnetic Field",
                      "NumberTitle", "off", "DefaultAxesFontSize", 14)
    hAxes1 = mplot.axes("unit", "normalized", "position", [0.07, 0.65, 0.72, 0.3],
                        "Box", "on")
    hAxes2 = mplot.axes("unit", "normalized", "position", [0.10, 0.25, 0.3, 0.3],
                        "Box", "on")
    hAxes3 = mplot.axes("unit", "normalized", "position", [0.50, 0.25, 0.3, 0.3],
                        "Box", "on")

    htextin = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "FontSize", 12, "position", [0.85, 0.96, 0.1, 0.03],
                              "max", 2, "string", "Input")
    htextr = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                             "backgroundcolor", [0.8, 0.8, 0.8],
                             "position", [0.85, 0.92, 0.04, 0.03], "max", 2,
                             "string", "r (m)")
    heditr = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                             "backgroundcolor", [1, 1, 1],
                             "position", [0.89, 0.92, 0.06, 0.03],
                             "string", "0.0")
    htextz = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                             "backgroundcolor", [0.8, 0.8, 0.8],
                             "position", [0.85, 0.88, 0.04, 0.03], "max", 2,
                             "string", "z (m)")
    heditz = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                             "backgroundcolor", [1, 1, 1],
                             "position", [0.89, 0.88, 0.06, 0.03],
                             "string", "0.0")
    htextrw = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "backgroundcolor", [0.8, 0.8, 0.8],
                              "position", [0.85, 0.84, 0.04, 0.03], "max", 2,
                              "string", "rmax")
    heditrw = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                              "backgroundcolor", [1, 1, 1],
                              "position", [0.89, 0.84, 0.06, 0.03],
                              "string", "0.5")
    htextzw = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "backgroundcolor", [0.8, 0.8, 0.8],
                              "position", [0.85, 0.8, 0.04, 0.03], "max", 2,
                              "string", "zmax")
    heditzw = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                              "backgroundcolor", [1, 1, 1],
                              "position", [0.89, 0.8, 0.06, 0.03],
                              "string", "1.5")

    htextout = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                               "FontSize", 12, "position", [0.85, 0.76, 0.1, 0.03],
                               "max", 2, "string", "Output")
    htextBr = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "backgroundcolor", [0.8, 0.8, 0.8],
                              "position", [0.85, 0.73, 0.04, 0.03], "max", 2,
                              "string", "Br(T)")
    heditBr = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                              "backgroundcolor", [1, 1, 1],
                              "position", [0.89, 0.73, 0.06, 0.03],
                              "string", "0.0")
    htextBz = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "backgroundcolor", [0.8, 0.8, 0.8],
                              "position", [0.85, 0.69, 0.04, 0.03], "max", 2,
                              "string", "Bz(T)")
    heditBz = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                              "backgroundcolor", [1, 1, 1],
                              "position", [0.89, 0.69, 0.06, 0.03],
                              "string", "0.0")
    htextB = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                             "backgroundcolor", [0.8, 0.8, 0.8],
                             "position", [0.85, 0.65, 0.04, 0.03], "max", 2,
                             "string", "B(T)")
    heditB = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                             "backgroundcolor", [1, 1, 1],
                             "position", [0.89, 0.65, 0.06, 0.03],
                             "string", "0.0")
    htextpsi = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                               "backgroundcolor", [0.8, 0.8, 0.8],
                               "position", [0.85, 0.61, 0.04, 0.03], "max", 2,
                               "string", "psi(Wb)")
    heditpsi = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                               "backgroundcolor", [1, 1, 1],
                               "position", [0.89, 0.61, 0.06, 0.03],
                               "string", "0.0")

    hpopimethod = mplot.uicontrol(hf, "unit", "normalized", "style", "popup",
                                  "position", [0.85, 0.52, 0.11, 0.05],
                                  "String", ["method 1", "method 2", "method 3"])
    htextrl = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "backgroundcolor", [0.8, 0.8, 0.8],
                              "position", [0.03, 0.12, 0.05, 0.04], "max", 2,
                              "string", "r_l (m)")
    heditrl = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                              "backgroundcolor", [1, 1, 1],
                              "position", [0.08, 0.12, 0.25, 0.05],
                              "string", "0.205,0.205,0.034,0.034,0.205,0.205")

    htextzl = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "backgroundcolor", [0.8, 0.8, 0.8],
                              "position", [0.34, 0.12, 0.07, 0.04], "max", 2,
                              "string", "z_l (m)")
    heditzl = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                              "backgroundcolor", [1, 1, 1],
                              "position", [0.41, 0.12, 0.25, 0.05],
                              "string", "-1.50625,-0.73625,-0.57825,0.57825,0.73625,1.50625")
    # %'-1.50625,-0.73625,-0.57825,0.57825,0.73625,1.50625'

    htexthl = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "backgroundcolor", [0.8, 0.8, 0.8],
                              "position", [0.03, 0.05, 0.05, 0.04], "max", 2,
                              "string", "h_l (m)")
    hedithl = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                              "backgroundcolor", [1, 1, 1],
                              "position", [0.08, 0.05, 0.25, 0.05],
                              "string", "0.09,0.09,0.038,0.038,0.09,0.09")

    htextwl = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "backgroundcolor", [0.8, 0.8, 0.8],
                              "position", [0.34, 0.05, 0.07, 0.04], "max", 2,
                              "string", "w_l (m)")
    heditwl = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                              "backgroundcolor", [1, 1, 1],
                              "position", [0.41, 0.05, 0.25, 0.05],
                              "string", "0.25,0.25,0.03,0.03,0.25,0.25")

    htextIl = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "backgroundcolor", [0.8, 0.8, 0.8],
                              "position", [0.67, 0.12, 0.05, 0.04], "max", 2,
                              "string", "I_l (A)")
    heditIl = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                              "backgroundcolor", [1, 1, 1],
                              "position", [0.72, 0.12, 0.25, 0.05],
                              "string", "80,80,80,80,80,80")

    htextNl = mplot.uicontrol(hf, "unit", "normalized", "style", "text",
                              "backgroundcolor", [0.8, 0.8, 0.8],
                              "position", [0.67, 0.05, 0.05, 0.04], "max", 2,
                              "string", "N_l (#)")
    heditNl = mplot.uicontrol(hf, "unit", "normalized", "style", "edit",
                              "backgroundcolor", [1, 1, 1],
                              "position", [0.72, 0.05, 0.25, 0.05],
                              "string", "510,820,90,90,820,510")

    hbloaddat = mplot.uicontrol(hf, "Style", "pushbutton", "unit", "normalized",
                                "Position", [0.85, 0.47, 0.1, 0.05],
                                "String", "Load Data", "Callback", loaddat_callback)
    hbsavedat = mplot.uicontrol(hf, "Style", "pushbutton", "unit", "normalized",
                                "Position", [0.85, 0.42, 0.1, 0.05],
                                "String", "Save Data", "Callback", savedat_callback)

    hbrun = mplot.uicontrol(hf, "Style", "pushbutton", "unit", "normalized",
                            "Position", [0.85, 0.3, 0.1, 0.05],
                            "String", "Run", "Callback", run_callback)
    hbsavefig = mplot.uicontrol(hf, "Style", "pushbutton", "unit", "normalized",
                                "Position", [0.85, 0.25, 0.1, 0.05],
                                "String", "Save Fig", "Callback", savefig_callback)

    hbexit = mplot.uicontrol(hf, "style", "push", "unit", "normalized",
                             "position", [0.85, 0.2, 0.1, 0.05],
                             "callback", "close", "string", "Exit")

    for name, value in (("hf", hf), ("hAxes1", hAxes1), ("hAxes2", hAxes2),
                        ("hAxes3", hAxes3), ("htextin", htextin), ("htextr", htextr),
                        ("heditr", heditr), ("htextz", htextz), ("heditz", heditz),
                        ("htextrw", htextrw), ("heditrw", heditrw),
                        ("htextzw", htextzw), ("heditzw", heditzw),
                        ("htextout", htextout), ("htextBr", htextBr),
                        ("heditBr", heditBr), ("htextBz", htextBz),
                        ("heditBz", heditBz), ("htextB", htextB),
                        ("heditB", heditB), ("htextpsi", htextpsi),
                        ("heditpsi", heditpsi), ("hpopimethod", hpopimethod),
                        ("htextrl", htextrl), ("heditrl", heditrl),
                        ("htextzl", htextzl), ("heditzl", heditzl),
                        ("htexthl", htexthl), ("hedithl", hedithl),
                        ("htextwl", htextwl), ("heditwl", heditwl),
                        ("htextIl", htextIl), ("heditIl", heditIl),
                        ("htextNl", htextNl), ("heditNl", heditNl),
                        ("hbloaddat", hbloaddat), ("hbsavedat", hbsavedat),
                        ("hbrun", hbrun), ("hbsavefig", hbsavefig),
                        ("hbexit", hbexit), ("run", run_callback),
                        ("savefig", savefig_callback),
                        ("loaddat", loaddat_callback),
                        ("savedat", savedat_callback)):
        ws[name] = value
