"""
This module contains utility functions pertaining to project 3
in FYS-STK4155.
"""

def my_figsize(column=True, subplots=(1, 1), ratio=None):
  """
  Specifies figure dimensions best suitable for latex.
  Credit to Johan Carlsen.
  """
  if column: 
    width_pt = 255.46837
  else:
    #width of latex text
    width_pt = 528.93675
  inch_per_pt = 1/72.27
  fig_ratio = (5**0.5 - 1)/2
  fig_width = width_pt*inch_per_pt
  fig_height = fig_width*fig_ratio*subplots[0]/subplots[1]
  fig_dim = (fig_width,fig_height)
  return fig_dim

