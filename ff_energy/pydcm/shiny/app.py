# from shiny import App, render, ui, reactive
# import ase
# from ase.visualize import view
# from htmltools import HTML
# from pathlib import Path, PosixPath
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import os
# import plotly.graph_objects as go
#
# from shiny import ui, App
# from shinywidgets import output_widget, render_widget
# import plotly.express as px
# import plotly.graph_objs as go
#
# from utils import set_axes_equal, get_data
#
# scan_data = pd.read_csv("../meoh_pbe0_adz_pc.csv")
# angles = list(scan_data["a2"])
# opt_rmses = np.array([[float(open("../logs/" + _).readlines()[i - 1]) for i, x in
#                        enumerate(open("../logs/" + _).readlines())
#                        if x.__contains__("RMSD")] for _ in os.listdir("../logs") if
#                       _.endswith("opt_fmdcm.log")])
#
# #  location where the cube files are stored
# data_path = Path("/data/unibas/boittier/pydcm_data/meoh_pbe0dz")
#
# mdcm_xyz = "/home/unibas/boittier/methanol_kern/refined.xyz"
# mdcm_format = "gaussian_{}_meoh_pbe0_adz_esp.cube.mdcm.xyz"
# cube_format = "gaussian_{}_meoh_pbe0_adz_esp.cube"
#
#
# def standardize(x):
#     return (x - min(x)) / (max(x) - min(x))
#
#
# X, ids, lcs = get_data()
#
# keys = ["lx", "ly", "lz", "ci", "angle", "N"]
# x = []
# y = []
# z = []
# ci = []
# angle = []
# N = []
# r1 = []
# r2 = []
# Erel = []
# r1s = list(scan_data["r1"])
# r2s = list(scan_data["r2"])
# Erels = list(scan_data["Erel"])
#
# for j in range(180):
#     for i in range(10):
#         ii = i
#         i = i * 3
#         x.append(lcs[j][i])
#         y.append(lcs[j][i + 1])
#         z.append(lcs[j][i + 2])
#         ci.append(ii)
#         angle.append(angles[j])
#         N.append(j)
#         Erel.append(Erels[j])
#         r1.append(r1s[j])
#         r2.append(r2s[j])
#
# df = pd.DataFrame({"lx": x, "ly": y, "lz": z,
#                    "r1": r1, "r2": r2, 'a2': angle,
#                    "ci": ci, "angle": angle,
#                    "Erel": Erel, "N": N})
#
# df["mag"] = np.sqrt((df["lx"]) ** 2 + (df["ly"]) ** 2 + (df["lz"]) ** 2)
#
# shifts = []
#
#
# def get_ase_view(A):
#     # load the mdcms
#     p1 = str(data_path / mdcm_format.format(A))
#     atoms1 = ase.io.read(p1)
#     # load the cube file
#     p2 = str(data_path / cube_format.format(A))
#     atoms2 = ase.io.read(p2)
#
#     # join the atoms
#     a = atoms1 + atoms2
#
#     html_ = view(a, viewer="x3d").data
#     html_ = html_.replace("0.96", "0.10")
#     html_ = html_.replace("0.76", "0.30")
#     html_ = html_.replace("0.66", "0.30")
#     html_ = html_.replace("0.31", "0.30")
#     html_ = html_.replace("0.20", "0.10")
#
#     return html_
#
#
# app_ui = ui.page_fluid(
#     ui.panel_title("MDCM"),
#
#     ui.layout_sidebar(
#
#         ui.panel_main(
#
#             ui.row(
#                 # ui.output_plot("energies", width="500px", height="500px"),
#
#                 ui.h2("Relative Energies"),
#                 output_widget("threedplot"),
#                 ui.h2("Local Charge Positions"),
#                 output_widget("positions"),
#             )
#         ),
#
#         ui.panel_sidebar(
#             ui.h2("Select conformer"),
#             ui.input_slider("n", "N", 0, 179, 90),
#             # ui.output_text_verbatim("txt"),
#             ui.input_action_button("add", "view the structure"),
#             ui.output_ui("moreControls", width="200px", height="200px"),
#             ui.input_checkbox_group(
#                 "colorkey", "Color key:", {k: k for k in df.keys()}
#             ),
#             # output_widget("my_widget")
#         ),
#     ),
# )
#
#
# def server(input, output, session):
#     @output
#     @render.text
#     def txt():
#         vhtml = (data_path / mdcm_format.format(input.n()))
#         return f"{lcs[0]} {vhtml}"
#
#     @output
#     @render.ui
#     @reactive.event(input.add, ignore_none=False)
#     def moreControls():
#         return ui.TagList(
#             ui.HTML(get_ase_view(input.n()))
#         )
#
#     @output
#     @render_widget
#     @reactive.event(input.add, ignore_none=False)
#     @reactive.event(input.colorkey, ignore_none=False)
#     def threedplot():
#         ckey = input.colorkey()
#         if len(ckey) < 1:
#             ckey = "angle"
#         else:
#             ckey = ckey[0]
#         fig = px.scatter_3d(
#             df, x='r1', y='r2', z='a2', color=ckey
#             # marginal="rug"
#         )
#         fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#         return fig
#
#     @output
#     @render_widget
#     @reactive.event(input.add, ignore_none=False)
#     @reactive.event(input.colorkey, ignore_none=False)
#     def positions(N=80, ckey="angle"):
#         # return make_pos_plot(input.n())
#         ckey = input.colorkey()
#         if len(ckey) < 1:
#             ckey = "angle"
#         else:
#             ckey = ckey[0]
#         LEN = len(df.lx.values)
#         fig = go.Figure(data=go.Scatter3d(
#             x=[df.lx.values[i // 2] if i % 2 else 0 for i in
#                range(2 * LEN)],
#             y=[df.ly.values[i // 2] if i % 2 else 0 for i in
#                range(2 * LEN)],
#             z=[df.lz.values[i // 2] if i % 2 else 0 for i in
#                range(2 * LEN)],
#             marker=dict(
#                 size=4,
#                 color=[df[ckey].values[i // 2] for i in range(LEN * 2)],
#             ),
#             line=dict(
#                 color=[df[ckey].values[i // 2] for i in range(LEN * 2)],
#                 width=2
#             )
#         ))
#         fig.update_layout(
#             scene=dict(
#                 xaxis=dict(nticks=3, range=[-2.5, 2.5], ),
#                 yaxis=dict(nticks=3, range=[-2.5, 2.5], ),
#                 zaxis=dict(nticks=3, range=[-2.5, 2.5], ),
#             ),
#             # width=700,
#             margin=dict(l=0, r=0, b=0, t=0)
#         )
#
#         fig.update_layout(scene_aspectmode='cube')
#
#         # fig.layout.height = 500
#         # fig.layout.width = 200
#         return fig
#
#
# app = App(app_ui, server)
#
