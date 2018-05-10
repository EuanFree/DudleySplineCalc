# -*- coding: utf-8 -*-
"""Application wrapper for the Dudley Method Spline Durability Calculation
Created on Tue Mar 27 14:55:48 2018

.. codeauthor:: Euan Freeman <euan.freeman@coxpowertrain.com>

"""

import wx
from StandardCalcs.SplineDurabilityDudleyMethod import DMSViewer

def launch():
    """Basic function to launch the application
    
    """
    
    appBE = wx.App()
    dmsView = DMSViewer()
    dmsView.Show()
    appBE.MainLoop()
    
#end def
    
if __name__ == '__main__':
    launch()
#end if