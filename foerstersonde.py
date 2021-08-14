import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
import altair as alt
import pint

# import mymodel as m
import matplotlib.pyplot as plt


u = pint.UnitRegistry()
# https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

# Länge, Breite und Höhe
a = 30 * u.millimeter
b = 1 * u.millimeter
c = 0.046 * u.millimeter

# Querschnitt des Kerns und Windungszahl der Spule
A_Q = b * c
N_W = 1000
H_0k1 = 0.575 * (u.ampere / u.centimeter)
B_0 = 0.37 * u.tesla

# Koeffizienten der Magnetisierungsskurve:
a_1 = 0.801
a_3 = 0.034
c_0 = 1.488
c_1 = 0.003
d = 2.8
h = np.linspace(-12, 12, 101)

# Kennwerte
H_refmax_slider = st.sidebar.slider(
    "Value for H_refmax:", value=1.0, min_value=0.0, max_value=5.0, step=0.05
)
H_ext_slider = st.sidebar.slider(
    "Value for H_ext:", value=1.0, min_value=0.0, max_value=5.0, step=0.05
)
H_refmax = H_refmax_slider * (u.ampere / u.meter)
H_ext = H_ext_slider * (u.ampere / u.meter)
f = 17 * (u.kilohertz)
omega = 2 * np.pi * f
m_r = 100000
m_rn = 100000
m_0 = 4 * np.pi * 1e-7 * (u.newton / u.ampere ** 2)
B_sat = (B_0 * np.pi) / 2
t = np.linspace(0, 0.15, 151) * u.millisecond

# Referenzfelder der beiden Spulen
H_ref1 = H_refmax * np.sin(omega * t)
H_ref2 = H_refmax * np.sin(omega * t + np.pi)

# Entmagnetisierungsfaktor
N = ((b * c) / a ** 2) * (np.log((4 * a) / (b + c) - 1))
H_Stern = (2 / np.pi) * (B_sat * (1 + N * (m_rn - 1))) / (m_rn * m_0)
H_Stern.ito(u.ampere / u.meter)  # EInheit schöner darstellen

# Innere Felder der Spulen
H_int1 = (H_ext + H_ref1) / (1 + N * (m_r - 1))
H_int2 = (H_ext + H_ref2) / (1 + N * (m_r - 1))

# normalisierte innere Felder
h_int1 = H_int1 / H_Stern
h_int2 = H_int2 / H_Stern

print(H_int1)
print(H_refmax)
print(h_int1)

d_slider = st.sidebar.slider(
    "Value for d:", value=d, min_value=0.0, max_value=5.0, step=0.05
)
# st.write("Slider number:", d_slider)


@st.cache
def b_kurve(a_1, a_3, c_0, c_1, d, h):
    arr_out = []
    out = []
    for i in range(len(h)):
        if np.amin(h) <= h[i] < -d:
            out = c_0 + c_1 * h[i]
        elif -d <= h[i] <= d:
            out = -(a_1 * h[i] - a_3 * h[i] ** 3)
        elif d < h[i] <= np.amax(h):
            out = -(c_0 - c_1 * h[i])
        arr_out = np.append(out, arr_out)
    return arr_out


def get_mag_curve(a_1, a_3, c_0, c_1, d, h):
    b = b_kurve(a_1, a_3, c_0, c_1, d, h)
    dataset = pd.DataFrame({"h": h, "b(h)": b}, columns=["h", "b(h)"])
    return dataset


df_mag_curve = get_mag_curve(a_1, a_3, c_0, c_1, d_slider, h)
df_b1 = get_mag_curve(a_1, a_3, c_0, c_1, d_slider, h_int1.magnitude)
df_b1["t"] = t.magnitude.tolist()
df_b2 = get_mag_curve(a_1, a_3, c_0, c_1, d_slider, h_int2.magnitude)
df_b2["t"] = t.magnitude.tolist()


"""
# Förstersonde
In diesem Sheet wird die Förstersonde berechnet. 

Im Detail:

* Magnetisierungskurve
* Magnetischer Fluss
* Harmonische Komponenten


"""

# Interactve Legend
# with st.echo(code_location='false'):
import altair as alt

selection = alt.selection_multi(fields=["b(h)"], bind="legend")
st.write(
    alt.Chart(df_mag_curve, title="Magnetisierungskennlinie")
    .mark_line()
    .transform_fold(fold=["b(h)"], as_=["variable", "value"])
    .encode(x="h:Q", y="b(h):Q", color="variable:N")
    .configure_axis(grid=True)
    .configure_view(strokeWidth=0)  #  .interactive()
)

"""
## Magnetischer Fluss
"""
import altair as alt

selection = alt.selection_multi(fields=["b(t)"], bind="legend")
st.write(
    alt.Chart(df_b1, title="Magnetischer FLuss")
    .mark_line()
    .transform_fold(fold=["b(t)"], as_=["variable", "value"])
    .encode(x="t:Q", y="b(h):Q", color="variable:N")
    .configure_axis(grid=True)
    .configure_view(strokeWidth=0)  #  .interactive()
)


print(h_int1)


# """
# ## Plotly example
# """
#
# with st.echo(code_location='below'):
#     import plotly.express as px
#
#     fig = px.scatter(
#         x=df_mag_curve["h"],
#         y=df_mag_curve["b(h)"],
#     )
#     fig.update_layout(
#         xaxis_title="h",
#         yaxis_title="b(h)",
#     )
#
#     st.write(fig)
#
#
# """
# ## Matplotlib example
# """
#
# with st.echo(code_location='below'):
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#
#     ax.scatter(
#         df_mag_curve["h"],
#         df_mag_curve["b(h)"],
#     )
#
#     ax.set_xlabel("h")
#     ax.set_ylabel("b(h)")
#
#     st.write(fig)
#
#
# """
# ## Bokeh example
# """
#
# with st.echo(code_location='below'):
#     from bokeh.plotting import figure
#
#     p = figure(
#         title="Simple line example",
#         x_axis_label='df["h"]',
#         y_axis_label='df["b(h)"]')
#
#     p.line(x=df_mag_curve["h"],
#              y=df_mag_curve["b(h)"],
#              legend_label="Temp.",
#              line_width=2)
#
#     st.bokeh_chart(p)
#     # st.write(p)


# #Learnings
# # Einheit und Wert
# print(a.magnitude)
# print(a.units)
#
# # Neue Variable in anderer Einheit
# a_m = a.to(u.meter)
# print(a_m)
# print(a)
# # Neue Variable, alte Variable wird überschrieben
# a_km = a.ito(u.kilometer)
# print('Print a: ' + str(a))
#
# #Baseunit und definierte Units
# print(a.to_base_units())
# print(a.to(u.inches))

# Unterschiedliche Visualisierungen
# https://share.streamlit.io/discdiver/data-viz-streamlit/main/app.py


# st.markdown("""
# #Sjajkjles models
# Below arek our sales predictions for this customer.
# """)
#
# st.latex(r'''
# a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
# \sum_{k=0}^{n-1} ar^k =
# a \left(\frac{1-r^{n}}{1-r}\right)
# ''')
#
# windows = st.slider("Forecast")
#
# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )
