# -*- coding: utf-8 -*-
"""Spline durability/Load capacity capability

Created on Fri Mar 16 11:48:46 2018

.. codeauthor:: Euan Freeman <euan.freeman@coxpowertrain.com>

"""

import numpy as np
from BaseCalculations.UnitConverter import PressureConverter,\
                                            LengthConverter
from nose.tools import assert_almost_equals, assert_equals

class DudleyMethodSpline(object):
    """Calculator of the spline and hub durability based upon the procedure \
    laid out in \"When Splines Need Stress Control\" by Darel W. Dudley
    
    :param t: Torque, :math:`T` [:math:`Nm`]
    :param dRE: Root diameter :math:`D_{re}` [:math:`m`]
    :param dH: Inner shaft diameter, :math:`D_h` [:math:`m`]
    :param supplyShockType: Number representing a specific supply shock type
    :param loadShockType: Number representing a specific load type
    :param n: Shaft speed [rev/min]
    :param nCyc: Shaft life time in terms of start/stop loading [cycles]
    :param nTotal: Total shaft life rotations [rev]
    :param reversible: Is the direction of rotation reversed?
    :param hardness: The measured hardness of the material
    :param hType: The type of hardness measurement used
    :param d: Pitch diameter, :math:`D` [:math:`m`]
    :param z: Number of spline teeth, :math:`z` []
    :param fE: Effective face width, :math:`F_e`, [:math:`m`]
    :param tC: Chordal thickness at pitchline, :math:`t_c` [:math:`m`]
    :param relativeMisalignment: Relative misalignment of the spline to the\
    hub based on misalignment/diameter
    :param h: Radial height of tooth in contact, :math:`H` [:math:`m`]
    :param phi: Pressure angle, :math:`\phi` [:math:`^{\circ}`]
    :param tW: Wall thickness, :math:`t_w` [:math:`m`]
    :param f: Full face width, :math:`F` [:math:`m`]
    :param dOi: Outside diameter of internally tootehd part, :math:`d_{oi}`\
    [:math:`'m`]
    :param dRi: Major diameter of internally tootehd part, :math:`d_{oi}`\
    [:math:`'m`]
    :param autoCalc: Whether to automatically calculate all the dudley factors
    :param y: Lewis form factor, :math:`Y` []
    :param toothEnd: Design of the tooth end, Straight or Crowned
    :param flexible: Whether the system has a flexible spline
    
    :type t: float
    :type dRE: float
    :type dH: float
    :type supplyShockType: int
    :type loadShockType: int
    :type n:int
    :type nCyc: int
    :type nTotal: int
    :type reversible: boolean
    :type hardness: float
    :type hType: str
    :type d: float
    :type z: int
    :type fE: float
    :type tC: float
    :type h: float
    :type phi: float
    :type tW: float
    :type f: float
    :type dOi: float
    :type dRi: float
    :type autoCalc: boolean
    :type y: float
    :type toothEnd: str
    
    """
    
    def __init__(self,\
                 t,\
                 dRe,\
                 supplyShockType,\
                 loadShockType,\
                 n,\
                 nCyc,\
                 nTotal,\
                 hardness,\
                 d,\
                 z,\
                 fE,\
                 tC,\
                 relativeMisalignment,\
                 h,\
                 phi,\
                 tW,\
                 f,\
                 dOi,\
                 dRi,\
                 autoCalc=True,\
                 dH=0.0,\
                 hType='Brinell',\
                 reversible=False,\
                 toothEnd='Straight',\
                 flexible=True,\
                 y=1.5):
        """Constructor
        
        """
        self._t = t
        self._dRe = dRe
        self._supplyShockType = supplyShockType
        self._loadShockType = loadShockType
        self._n = n
        self._nCyc = nCyc
        self._nTotal = nTotal
        self._hardness = hardness
        self._d = d
        self._z = z
        self._fE = fE
        self._tC = tC
        self._relativeMisalignment = relativeMisalignment
        self._h = h
        self._phi = phi
        self._tW = tW
        self._f = f
        self._dOi = dOi
        self._dRi = dRi
        self._dH = dH
        self._hType = hType
        self._reversible = reversible
        self._toothEnd = toothEnd
        self._flexible = flexible
        self._y = y
        if autoCalc:
            self.calculate()
        #end if
    #end def
    
    def calculate(self):
        """Based on the stored variables calculate the Dudley method
        
        """
        self._shaftStress = self.solidShaftStress(self._t,self._dRe) \
            if self._dH == 0.0 else self.hollowShaftStress(self._t,\
                                                           self._dRe,\
                                                           self._dH)
        self._appFactor = \
            self.getSplineApplicationFactor(self._supplyShockType,\
                                            self._loadShockType)
        self._splineLF = self.getLifeFactor(self._nCyc,\
                                            reversible=self._reversible)
        self._allowShaftStress = \
            self.getAllowableShearStressByHardness(self._hardness,\
                                                   hType=self._hType)
        self._maxShaftStress = self.maximumShaftStress(self._shaftStress,\
                                                       self._appFactor,\
                                                       self._splineLF)
        self._shaftSafetyFactor = \
            self.getShaftSafetyFactor(self._maxShaftStress,\
                    self._allowShaftStress)
        self._teethLoadkM = \
            self.getLoadDistributionFactorSpline(self._relativeMisalignment,\
                                                 self._fE)
        self._teethShearStress = self.getTeethShearStress(self._t,\
                                                          self._teethLoadkM,\
                                                          self._d,\
                                                          self._z,\
                                                          self._fE,\
                                                          self._tC)
        self._maxTeethShearStress = \
            self.maximumShaftStress(self._teethShearStress,\
                                 self._appFactor,\
                                 self._splineLF)
        self._teethSafetyFactor = \
            self.getShaftSafetyFactor(self._maxTeethShearStress,\
                                      self._allowShaftStress)
        self._compStress = self.getCompressiveStress(self._t,\
                                                     self._teethLoadkM,\
                                                     self._d,\
                                                     self._z,\
                                                     self._fE,\
                                                     self._h)
        self._allowCompStress = \
            self.getAllowableCompressiveStressForSplines(self._hardness,\
                                                 self._hType,\
                                                 toothEnd=self._toothEnd)
        if self._flexible:
            self._lifeWear = self.getSplineWearLifeFactor(self._nTotal)
            self._factoredCompStress = \
                self.getAllowableCompressiveStress(self._compStress,\
                                                   self._appFactor,\
                                                   self._lifeWear)
        else:
            self._factoredCompStress = \
                self.getAllowableCompressiveStress(self._compStress,\
                                                   self._appFactor,\
                                                   self._splineLF,\
                                                   flexible=False)
        #end if
        self._compSafetyFactor = self._allowCompStress/self._factoredCompStress
        self._burstRad = self.getBurstingRadialStress(self._t,\
                                                      self._phi,\
                                                      self._d,\
                                                      self._tW,\
                                                      self._f)
        self._burstCentrifugal = \
            self.getBurstingCentrifugalStress(self._n,\
                                              self._dOi,\
                                              self._dRi)
        self._burstTens = self.getBurstingTensileStress(self._t,\
                                                        self._d,\
                                                        self._fE,\
                                                        y=self._y)
        self._burstTotal = self.getTotalBurstingStress(self._appFactor,\
                                                       self._teethLoadkM,\
                                                       self._burstRad,\
                                                       self._burstCentrifugal,\
                                                       self._burstTens)
        self._allowBurstStress = \
            self.getAllowableBurstingStressByHardness(self._hardness,\
                                                      hType=self._hType)
        self._burstSafetyFactor = \
            self.getBurstingSafetyFactor(self._burstTotal,\
                                         self._allowBurstStress,\
                                         self._splineLF)
    #end def
    
    def solidShaftStress(self,t,dRE):
        """Calculate the shaft stress for a solid shaft, :math:`S_s`
        
        :param t: Shaft torque, :math:`T` [:math:`Nm`]
        :param dRE: Root diameter :math:`D_{re}` [:math:`m`]
        
        :type t: float
        :type dRE: float
        
        :returns: Shaft stress, :math:`S_s` [:math:`N/m^2`]
        :rtype: float
        
        """
        return 16.*t/(np.pi*dRE**3.)
    #end def
    
    def hollowShaftStress(self,t,dRE,dH):
        """Calculate the shaft stress for a hollow shaft, :math:`S_s`
        
        :param t: Shaft torque, :math:`T` [:math:`Nm`]
        :param dRE: Root diameter :math:`D_{re}` [:math:`m`]
        :param dH: Inner shaft diameter, :math:`D_h` [:math:`m`]
        
        :type t: float
        :type dRE: float
        
        :returns: Shaft stress, :math:`S_s` [:math:`N/m^2`]
        :rtype: float
        
        """
        return 16.*t*dRE/(np.pi*(dRE**4.-dH**4.))
    #end def
    
    def getSplineApplicationFactor(self,supplyShockType,loadShockType):
        """Get the spline application factor, :math:`K_a` based on look up \
        table
        
        Look up the spline application factor based on the shock level of both
        the supply and delivery of power through the spline, essentially on
        either side of the spline and hub
        
        +---------------+-----------------------------------------------------+
        | Power Source  | Type of Load                                        |
        |               +-------------+-------------+------------+------------+
        |               | Uniform     | Light Shock | Medium     | Heavy shock|
        |               | shock       |             | shock      |            |
        |               | (generator, | (pulsing    | (actuating | (punches,  |
        |               | fan)        | pump)       | pump)      | shears)    |
        +---------------+-------------+-------------+------------+------------+
        | Uniform       | 1.0         | 1.2         | 1.5        | 1.8        |
        | (turbine)     |             |             |            |            |
        +---------------+-------------+-------------+------------+------------+
        | Light Shock   | 1.2         | 1.3         | 1.8        | 2.1        |
        | (hydr. motor) |             |             |            |            |
        +---------------+-------------+-------------+------------+------------+
        | Medium Shock  | 2.0         | 2.2         | 2.4        | 2.8        |
        | (ICE)         |             |             |            |            |
        +---------------+-------------+-------------+------------+------------+
        
        :param supplyShockType: Number representing a specific supply shock type
        :param loadShockType: Number representing a specific load type
        
        :type supplyShockType: int
        :type loadShockType: int
        
        Supply shock type
        ==================
        
        +---------------+--------------+
        | Type          | Number       |
        +---------------+--------------+
        | Uniform       | 0            |
        +---------------+--------------+
        | Light Shock   | 1            |
        +---------------+--------------+
        | Medium Shock  | 2            |
        +---------------+--------------+
        
        Load shock type
        ================
        
        +---------------+--------------+
        | Type          | Number       |
        +---------------+--------------+
        | Uniform       | 0            |
        +---------------+--------------+
        | Light Shock   | 1            |
        +---------------+--------------+
        | Medium Shock  | 2            |
        +---------------+--------------+
        | Heavy Shock   | 3            |
        +---------------+--------------+
        
        :returns: Returns the spline application factor, :math:`K_a` []
        :rtype: float
        """
        
        kTable = [[1.,1.2,1.5,1.8],[1.2,1.3,1.8,2.1],[2.,2.2,2.4,2.8]]
        return kTable[supplyShockType][loadShockType]
    #end def
    
    def getLifeFactor(self,nCyc,reversible=False):
        """Get the life factor, :math:`L_f` based on look up table
        
        +-------------------------+-------------------------------------+
        | Number of Torque Cycles | Life factor, :math:`L_f`            |
        |                         +-----------------+-------------------+
        |                         | Unidirectional  | Fully-reversed    |
        +-------------------------+-----------------+-------------------+
        | 1000                    | 1.8             | 1.8               |
        +-------------------------+-----------------+-------------------+
        | 10000                   | 1.0             | 1.0               |
        +-------------------------+-----------------+-------------------+
        | 100000                  | 0.5             | 0.4               |
        +-------------------------+-----------------+-------------------+
        | 1000000                 | 0.4             | 0.3               |
        +-------------------------+-----------------+-------------------+
        | 10000000                | 0.3             | 0.2               |
        +-------------------------+-----------------+-------------------+
        
        :param nCyc: Shaft life time [cycles]
        :param reversible: Is the direction of rotation reversed?
        
        :type nCyc: int
        :type reversible: boolean
        
        :returns: Returns :math:`L_f` []
        :rtype: float
        
        """
        
        if nCyc < 1000:
            return 1.8 if reversible else 1.8
        elif nCyc < 10000:
            return 1.0 if reversible else 1.0
        elif nCyc < 100000:
            return 0.4 if reversible else 0.5
        elif nCyc < 1000000:
            return 0.3 if reversible else 0.4
        else:
            return 0.2 if reversible else 0.3
        #end if
    #end def
    
    def maximumShaftStress(self,sS,kA,lF):
        """Get the maximum shaft stress, :math:`{S_s}^\prime`
        
        :param sS: Calculated Shaft stress, :math:`S_s` [:math:`N/m^2`]
        :param kA: Spline application factor, :math:`k_a` []
        :param lF: Life factorm :math:`L_f` []
        
        :type sS: float
        :type kA: float
        :type lF: float
        
        :returns: Returns the maximum shaft stress , :math:`{S_s}^\prime`[] \
        [:math:`N/{m^2}]
        :rtype: float
        
        """
        return sS*kA/lF
    #end def
    
    def getAllowableShearStressByHardness(self,hardness,hType='Brinell'):
        """Get the maximum allowable shear stress for a spline based upon the \
        the surface hardness of the component, :math:`{{S_s}^\prime}_{max}`
        
        :param hardness: The measured hardness of the material
        :param hType: The type of hardness measurement used
        
        :type hardness: float
        :type hType: str
        
        :returns: Returns the maximum allowable shear stress, \
        :math:`{{S_s}^\prime}_{max}` [:math:`N/{m^2}]
        :rtype: float
        
        Uses the following lookup table:
        
        +-------------------------+---------------------------+---------------+
        | Material                | Hardness                  | Max Allowable |
        |                         +-------------+-------------+ Stress        |
        |                         | Brinell     | Rockwell C  | [psi]         |
        +-------------------------+-------------+-------------+---------------+
        | Steel                   | 160-200     |             | 20000         |
        +-------------------------+-------------+-------------+---------------+
        | Steel                   | 230-260     |             | 30000         |
        +-------------------------+-------------+-------------+---------------+
        | Steel                   | 302-351     | 33-38       | 40000         |
        +-------------------------+-------------+-------------+---------------+
        | Thru-Hardened Steel     |             | 42-46       | 45000         |
        +-------------------------+-------------+-------------+---------------+
        | Surface Hardened Steel  |             | 48-53       | 40000         |
        +-------------------------+-------------+-------------+---------------+
        | Case Hardened Steel     |             | 58-63       | 50000         |
        +-------------------------+-------------+-------------+---------------+
        
        :returns: Returns the maximum allowable shear stress for a spline \
        based upon the the surface hardness of the component, \
        :math:`{{S_s}^\prime}_{max}`
        
        :rtype: float
        
        """
        pc = PressureConverter()
        if hType == 'Brinell':
            if hardness < 200.:
                return pc.psiToPa(20000)
            elif hardness < 260.:
                return pc.psiToPa(30000)
            elif hardness < 351.:
                return pc.psiToPa(40000)
            #end if
        elif hType == 'Rockwell C':
            if hardness < 38.:
                return pc.psiToPa(40000)
            elif hardness < 46.:
                return pc.psiToPa(45000)
            elif hardness < 53:
                return pc.psiToPa(40000)
            elif hardness < 63:
                return pc.psiToPa(50000)
        else:
            raise UnrecognisedHardnessTypeException()
        #end if
    #end def
    
    def getShaftSafetyFactor(self,maxShaftStress,allowableShaftStress):
        """Get the shaft safety factor, :math:`f_{safe}` based on the \
        calculated maximum shaft stress and maximum allowable shaft stress
        
        :param maxShaftStress: The maximum shaft stress, \
        :math:`{{S_s}^\prime}`[] [:math:`N/{m^2}]
        :param allowableShaftStress: The maximum allowable shear stress, \
        :math:`{{S_s}^\prime}_{max}` [:math:`N/{m^2}`]
        
        :type maxShaftStress: float
        :type allowableShaftStress: float
        
        :returns: Returns the shaft safety factor, :math:`f_{safe}`
        :rtype: float
        
        """
        
        return allowableShaftStress/maxShaftStress
    #end def
    
    def getTeethShearStress(self,t,kM,d,z,fE,tC):
        """Get the teeth shear stress, :math:`S_{s,j}`
        
        :param t: Torque, :math:`T` [:math:`Nm`]
        :param kM: Load distribution factor, :math:`K_m` []
        :param d: Pitch diameter, :math:`D` [:math:`m`]
        :param z: Number of spline teeth, :math:`z` []
        :param fE: Effective face width, :math:`F_e`, [:math:`m`]
        :param tC: Chordal thickness at pitchline, :math:`t_c` [:math:`m`]
        
        :type t: float
        :type kM: float
        :type d: float
        :type z: int
        :type fE: float
        :type tC: float
        
        :returns: Returns the teeth shear stress, :math:`S_{s,j}` \
        [:math:`N/{m^2}`]
        :rtype: float
        
        """
        return (4.*t*kM)/(d*z*fE*tC)
    #end def
    
    def getLoadDistributionFactorSpline(self,relativeMisalignment,fE):
        """Get the load distribution factor, :math:`K_m` based on the relative \
        misalignment and face width
        
        :param relativeMisalignment: Relative misalignment of the spline to the\
        hub based on misalignment/diameter
        :param fE: Face width, :math:`F_e` of the spline tooth [m]
        
        :type relativeMisalignment: float
        :type faceWidth: float
        
        :math:`K_m` looked up from table
        
        +--------------+------------------------------------------------------+
        | Misalignment | Factor :math:`K-m`                                   |
        |              +-------------+-------------+-------------+------------+
        |              | 12.7mm face | 25.4mm face | 50.8mm face | 101mm face |
        |              | width       | width       | width       | width      |
        +--------------+-------------+-------------+-------------+------------+
        | 0.001        | 1.0         | 1.0         | 1.0         | 1.5        |
        +--------------+-------------+-------------+-------------+------------+
        | 0.002        | 1.0         | 1.0         | 1.5         | 2.0        |
        +--------------+-------------+-------------+-------------+------------+
        | 0.004        | 1.0         | 1.5         | 2.0         | 2.5        |
        +--------------+-------------+-------------+-------------+------------+
        | 0.008        | 1.5         | 2.0         | 2.5         | 3.0        |
        +--------------+-------------+-------------+-------------+------------+
        
        :returns: Returns load distribution factor, :math:`K_m` []
        :rtype: float
        
        """
        
        lookUp = [[1.,1.,1.,1.5],[1.,1.,1.5,2.],[1.,1.5,2.,2.5],[1.5,2.,2.5,3.]]
        
        m = 0
        for i in [0.001,0.002,0.004,0.008]:
            if relativeMisalignment < i:
                break
            else:
                m += 1
            #end if
        #end for
        m = 3 if m > 3 else m
        n = 0
        for i in [12.7e-3,25.4e-3,50.8e-3,101.e-3]:
            if fE < i:
                break
            else:
                n += 1
            #end if
        #end for
        n = 3 if n > 3 else n
        return lookUp[m][n]
    #end def
    
    def getCompressiveStress(self,t,kM,d,z,fE,h):
        """Get the compressive stress, :math:`\sigma_c`, acting on the spline \
        teeth
        
        :param t: Torque, :math:`T` [:math:`Nm`]
        :param kM:  Load distribution factor, :math:`K_m` []
        :param d: Pitch diameter, :math:`D` [:math:`m`]
        :param z: Number of spline teeth, :math:`z` []
        :param fE: Effective face width, :math:`F_e`, [:math:`m`]
        :param h: Radial height of tooth in contact, :math:`H` [:math:`m`]
        
        :type t: float
        :type kM: float
        :type d: float
        :type z: int
        :type fE: float
        :type tC: float
        
        :returns: Returns the compressive stress, :math:`\sigma_c` \
        [:math:`N_{m^2}`]
        :rtype: float
        
        """
        return (2.*t*kM)/(d*z*fE*h)
    #end def
    
    def getAllowableCompressiveStressForSplines(self,\
                                                hardness,\
                                                hType='Brinell',
                                                toothEnd='Straight'):
        """Get the allowable compressive stress for the splines, \
        :math:`{S^{\prime}}_c`
        
        :param hardness: Hardness value, []
        :param hType: Harness measurement technique, Brinell or Rockwell C
        :param toothEnd: Design of the tooth end, Straight or Crowned
        
        :type hardness: float
        :type hType: str
        :type toothEnd: str
        
        Looked up from the following table
        
        +-------------------------+---------------------------+----------------+
        | Material                | Hardness                  | Max Allowable  |
        |                         |                           | Stress         |
        |                         +-------------+-------------+--------+-------+
        |                         | Brinell     | Rockwell C  |Straight|Crowned|
        +-------------------------+-------------+-------------+--------+-------+
        | Steel                   | 160-200     |             |  1500  | 6000  |
        +-------------------------+-------------+-------------+--------+-------+
        | Steel                   | 230-260     |             |  2000  | 8000  |
        +-------------------------+-------------+-------------+--------+-------+
        | Steel                   | 302-351     | 33-38       |  3000  | 12000 |
        +-------------------------+-------------+-------------+--------+-------+
        | Surface Hardened Steel  |             | 48-53       |  4000  | 16000 |
        +-------------------------+-------------+-------------+--------+-------+
        | Case Hardened Steel     |             | 58-63       |  5000  | 20000 |
        +-------------------------+-------------+-------------+--------+-------+
        
        
        :returns: Returns the allowable compressive stress for the splines, \
        :math:`{S^{\prime}}_c` [:math:`N/m^2`]
        :rtype: float
        
        """
        pc = PressureConverter()
        if toothEnd not in ['Straight','Crowned']:
            raise UnrecognisedToothEndException()
        #end if
        if hType == 'Brinell':
            if hardness < 200:
                if toothEnd == 'Straight':
                    return pc.psiToPa(1500)
                else:
                    return pc.psiToPa(6000)
                #end if
            elif hardness < 260:
                if toothEnd == 'Straight':
                    return pc.psiToPa(2000)
                else:
                    return pc.psiToPa(8000)
                #end if
            elif hardness < 351:
                if toothEnd == 'Straight':
                    return pc.psiToPa(3000)
                else:
                    return pc.psiToPa(12000)
                #end if
            #end if
        elif hType == 'Rockwell C':
            if hardness < 38:
                if toothEnd == 'Straight':
                    return pc.psiToPa(3000)
                else:
                    return pc.psiToPa(12000)
                #end if
            elif hardness < 53:
                if toothEnd == 'Straight':
                    return pc.psiToPa(4000)
                else:
                    return pc.psiToPa(16000)
                #end if
            elif hardness < 63:
                if toothEnd == 'Straight':
                    return pc.psiToPa(5000)
                else:
                    return pc.psiToPa(20000)
                #end if
            #end if
        else:
            raise UnrecognisedHardnessTypeException()
        #end if
    #end def
    
    def getSplineWearLifeFactor(self,nCyc):
        """Get the spline wear factor, :math:`L_w`
        
        :param nCyc: Number of cycles, :math:`n` []
        
        :type nCyc: float
        
        Get the life factor from the following table 
        
        +-------------------------+----------------------------+
        | No. of revolution       | Life factor, :math:`L_w`   |
        +-------------------------+----------------------------+
        | :math:`10^4`            | 4.0                        |
        +-------------------------+----------------------------+
        | :math:`10^5`            | 2.8                        |
        +-------------------------+----------------------------+
        | :math:`10^6`            | 2.0                        |
        +-------------------------+----------------------------+
        | :math:`10^7`            | 1.4                        |
        +-------------------------+----------------------------+
        | :math:`10^8`            | 1.0                        |
        +-------------------------+----------------------------+
        | :math:`10^9`            | 0.7                        |
        +-------------------------+----------------------------+
        | :math:`10^10`           | 0.5                        |
        +-------------------------+----------------------------+
        
        :returns: Returns the spline wear factor, :math:`L_w`
        :rtype: float
        
        """
        
        if nCyc < 1e4:
            return 4.0
        elif nCyc < 1e5:
            return 2.8
        elif nCyc < 1e6:
            return 2.
        elif nCyc < 1e7:
            return 1.4
        elif nCyc < 1e8:
            return 1.
        elif nCyc < 1e9:
            return 0.7
        elif nCyc < 1e10:
            return 0.5
        #end if
    #end def
    
    def getAllowableCompressiveStress(self,sC,kA,lW,flexible=True):
        """Get the allowable compressive stress, :math:`{S^{\prime}}_c`
        
        :param sC: Maximum compressive stress, :math:`S_c` [:math:`N/mm^2`]
        :param kA: Application factor, :math:`K_a` []
        :param lW: Life wear factor, :math:`L_w` [] or Life factor, \
        :math:`L_f` []
        :param flexible: Whether the spline teeth are flexible
        
        :type sC: float
        :type kA: float
        :type lW: float
        :type flexible: boolean
        
        :returns: Returns the allowable compressive stress, \
        :math:`{S^{\prime}}_c` [:math:`N/mm^2`]
        :rtype: float
        
        """
        if flexible:
            return sC*kA/lW
        else:
            return sC*kA/(9.*lW)
        #end if
    #end def
    
    def getBurstingRadialStress(self,t,phi,d,tW,f):
        """Get the bursting stress due to radial stress, :math:`S_1`
        
        :param t: Torque, :math:`T` [:math:`Nm`]
        :param phi: Pressure angle, :math:`\phi` [:math:`^{\circ}`]
        :param d: Pitch diameter, :math:`D` [:math:`m`]
        :param tW: Wall thickness, :math:`t_w` [:math:`m`]
        :param f: Full face width, :math:`F` [:math:`m`]
        
        :type t: float
        :type phi: float
        :type d: float
        :type tW: float
        :type f: float
        
        :returns: Returns the bursting radial stress, :math:`S_1` \
        [:math:`N/m^2`]
        :rtype: float
        
        """
        return t*np.tan(phi)/(np.pi*d*tW*f)
    #end def
    
    def getBurstingCentrifugalStress(self,n,dOi,dRi):
        """Get the bursting stress to the centrifugal stress, :math:`S_2`
        
        :param n: Rotational speed, :math:`n` [:math:`rpm`]
        :param dOi: Outside diameter of internally tootehd part, :math:`d_{oi}`\
        [:math:`'m`]
        :param dRi: Major diameter of internally tootehd part, :math:`d_{oi}`\
        [:math:`'m`]
        
        :type n: int
        :type dOi: float
        :type dRi: float        
        
        :returns: Returns the bursting stress due to the centrifugal stress, \
        :math:`S_2` [:math:`N/m^2`]
        :rtype: float
        
        """
        lc = LengthConverter()
        pc = PressureConverter()
        return pc.psiToPa(0.828*1.0e-6*float(n)**2.*\
                ((2.*lc.mToInch(dOi)**2.+(0.424*lc.mToInch(dRi)**2.))))
    #end def
    
    def getBurstingTensileStress(self,t,d,fE,y=1.5):
        """Get teh bursting stress due to the tensile stress, :math:`S_3`
        
        :param t: Torsion, :math:`T` [:math:`Nm`]
        :param d: Pitch diameter, :math:`D` [m]
        :param fS: Effective face area, :math:`F_e` [:math:`m^2`]
        :param y: Lewis form factor, :math:`Y` []
        
        :returns: Returns the bursting stress to the tensile stress, \
        :math:`S_3` [:math:`N/m^2`]
        :rtype: float
        
        """
        return (4.*t)/(d**2.*fE*y)
    #end def
    
    def getLewisFormFactor(self,phiNR,phiNL,hF,sF,k,kPsi=1.,cH=1.):
        """Get the Lewis form factor, :math:`Y`, using the calulation from \
        Eqn. 5.78 from AGMA 908-B89
        
        :param phiNR: Operating normal pressure angle, :math:`\Phi_{nr}` \
        [:math:`^{\circ}`]
        :param phiNL: Load angle, :math:`\Phi_{nL}` [:math:`^{\circ}`]
        :param hF: Height of Lewis parabola, :math:`h_F` [:math:`m`]
        :param sF: Tooth thickness at critcal point, :math:`s_F` [:math:`m`]
        :param kPsi: Helix angle factor, :math:`k_\psi` []
        :param cH: Helical factor, :math:`C_h` []
        
        :type phiNR: float
        :type phiNL: float
        :type hF:float
        
        Assumptions
        ------------
        
        1. Due to the spline acting like a spur gear, the helix angle factor \
        :math:`K_\psi` = 1.0 and the helical factor, :math:`C_h`, = 1.0
        
        :returns: Lewis form factor (known as the Tooth Form Factor in the \
        reference document), :math:`Y` []
        :rtype: float
        
        """
        return kPsi/((np.cos(phiNL)/np.cos(phiNR))*((6.*hF/(sF**2.*cH))-\
                      (np.tan(phiNL)/sF)))
    #end def
    
### TODO: Add in functionality to calculate the parameters required for the 
    #Lewis form factor 
    
    def getTotalBurstingStress(self,kA,kM,s1,s2,s3):
        """Get the total bursting stress,:math:`S_t`, on a tooth in the spline \
        based on the combination of various bursting stresses
        
        :param kA: Application factor, :math:`K_a` []
        :param kM: Load distribution factor, :math:`K_m` []
        :param s1: Bursting radial stress, :math:`S_1` [:math:`N/m^2`]
        :param s2: Bursting stress due to the centrifugal stress, \
        :math:`S_2` [:math:`N/m^2`]
        :param s3: Bursting stress due to the tensile stress, \
        :math:`S_3` [:math:`N/m^2`]
        
        :type kA: float
        :type kM: float
        :type s1: float
        :type s2: float
        :type s3: float
        
        :returns: Returns the total bursting stress,:math:`S_t` [:math:`N/m^2`]
        :rtype: float
        
        """
        return (kA*kM*(s1+s3))+s2
    #end def
    
    def getAllowableBurstingStressByHardness(self,hardness,hType='Brinell'):
        """Get the maximum allowable bursting stress for a spline based upon \
        the surface hardness of the component, :math:`{{S_t}^\prime}_{max}`
        
        :param hardness: The measured hardness of the material
        :param hType: The type of hardness measurement used
        
        :type hardness: float
        :type hType: str
        
        
        Uses the following lookup table:
        
        +-------------------------+---------------------------+---------------+
        | Material                | Hardness                  | Max Allowable |
        |                         +-------------+-------------| Stress        |
        |                         | Brinell     | Rockwell C  | [psi]         |
        +-------------------------+-------------+-------------+---------------+
        | Steel                   | 160-200     |             | 22000         |
        +-------------------------+-------------+-------------+---------------+
        | Steel                   | 230-260     |             | 32000         |
        +-------------------------+-------------+-------------+---------------+
        | Steel                   | 302-351     | 33-38       | 45000         |
        +-------------------------+-------------+-------------+---------------+
        | Thru-Hardened Steel     |             | 42-46       | 45000         |
        +-------------------------+-------------+-------------+---------------+
        | Surface Hardened Steel  |             | 48-53       | 50000         |
        +-------------------------+-------------+-------------+---------------+
        | Case Hardened Steel     |             | 58-63       | 55000         |
        +-------------------------+-------------+-------------+---------------+
        
        :returns: Returns the maximum allowable bursting stress, \
        :math:`{{S_t}^\prime}_{max}` [:math:`N/{m^2}]
        :rtype: float
        
        """
        pc = PressureConverter()
        if hType == 'Brinell':
            if hardness < 200.:
                return pc.psiToPa(22000)
            elif hardness < 260.:
                return pc.psiToPa(32000)
            elif hardness < 351.:
                return pc.psiToPa(45000)
            #end if
        elif hType == 'Rockwell C':
            if hardness < 46.:
                return pc.psiToPa(45000)
            elif hardness < 53:
                return pc.psiToPa(50000)
            elif hardness < 63:
                return pc.psiToPa(55000)
        else:
            raise UnrecognisedHardnessTypeException()
        #end if
    #end def
    
    def getBurstingSafetyFactor(self,sTTot,sTTotMax,lF):
        """Get the bursting safety factor, :math:`F_{max,burst}`
        
        :param sTTot:Total bursting stress, \
        :math:`{{S_t}^\prime}` [:math:`N/{m^2}]
        :param sTTotMax: Maximum allowable bursting stress, \
        :math:`{{S_t}^\prime}_{max}` [:math:`N/{m^2}]
        :param lF: Life factor, :math:`L_f`, []
        
        :type sTTot: float
        :type sTTotMax: float
        :type lF: float
        
        :return: Returns the bursting safety factor, :math:`F_{max,burst}`
        :rtype float
        
        """
        
        return sTTotMax/(sTTot/lF)
    #end def
    
    def __repr__(self):
        """Return a representation of the data contained within the object
        
        :returns: Returns string representation of the object contents
        :rtype: str
        
        """
        self.calculate()
        outStr = 'DudleyMethodSpline object\n\n'
        outStr += 'Shaft Stress Calculations\n\n'
        outStr += 'Shaft Stress: {0}MPa\n'.format(self._shaftStress/1.0e6)
        outStr += 'Allowable Shear Stress: {0}MPa\n\n'.\
            format(self._allowShaftStress/1.0e6)
        outStr += 'Shaft Stress - adjusted for application and life'
        outStr += ': {0}MPa\n'.format(self._maxShaftStress/1.0e6)
        outStr += 'Shaft Safety Factor: {0}\n\n'.format(self._shaftSafetyFactor)
        outStr += 'Teeth Shear Stress\n\n'
        outStr += 'Teeth Shear Stress: {0}MPa\n'.\
            format(self._teethShearStress/1.0e6)
        outStr += 'Teeth Max Stress: {0}MPa\n'.\
            format(self._maxTeethShearStress/1.0e6)
        outStr += 'Teeth Safety Factor: {0}\n\n'.format(self._teethSafetyFactor)
        outStr += 'Compressive Stress\n\n'
        outStr += 'Compressive Stress: {0}MPa\n'.format(self._compStress/1.0e6)
        outStr += 'Allowable Compressive Stress: {0}MPa\n'.\
            format(self._allowCompStress/1.0e6)
        outStr += 'Factored Compressive Stress: {0}MPa\n'.\
            format(self._factoredCompStress/1.0e6)
        outStr += 'Compressive Safety Factor: {0}\n\n'.\
            format(self._compSafetyFactor)
        outStr += 'Bursting Stress\n\n'
        outStr += 'Bursting Stress - Radial: {0}\n'.format(self._burstRad/1.0e6)
        outStr += 'Bursting Stress - Centrifugal: {0}\n'.\
            format(self._burstCentrifugal/1.0e6)
        outStr += 'Bursting Stress - Tensile: {0}\n'.\
            format(self._burstTens/1.0e6)
        outStr += 'Bursting Stress - Total: {0}\n'.\
            format(self._burstTotal/1.0e6)
        outStr += 'Allowable Bursting Stress: {0}MPa\n'.\
            format(self._allowBurstStress/1.0e6)
        outStr += 'Bursting Safety Factor: {0}'.format(self._burstSafetyFactor)
        outStr += '\n\nFactors\n\n'
        outStr += 'Spline Application Factor, Ka: {0}\n'.format(self._appFactor)
        outStr += 'Fatigue Life Factor, Lf: {0}\n'.format(self._splineLF)
        outStr += 'Load distibution Factor - Teeth: {0}\n'.\
            format(self._teethLoadkM)
        try:
            outStr += 'Flexible Life Factor, Lw: {0}\n'.\
                format(self._lifeWear)
        except AttributeError:
            pass
        #end if
        
        
        return outStr
    #end def
    
    def __str__(self):
        """Calls the representation method
        
        """
        return self.__repr__()
    #end def
    
### Test section - For SQA
    
    def test_SolidShaftStress(self):
        """Test the solid shaft stress calculation
        
        """
        assert_almost_equals(self.solidShaftStress(2.,3.),\
                             0.37725,\
                             places=4)
    #end def
    
    def test_HollowShaftStress(self):
        """Test the hollow shaft stress calculation
        
        """
        assert_almost_equals(self.hollowShaftStress(2.,3.,2.),\
                             0.470119,\
                             places=4)
    #end def
    
    def test_SplineApplicationFactor(self):
        """Test the selection of the application factor
        
        """
        assert_equals(self.getSplineApplicationFactor(0,0),1.0)
        assert_equals(self.getSplineApplicationFactor(1,0),1.2)
        assert_equals(self.getSplineApplicationFactor(2,0),2.0)
        assert_equals(self.getSplineApplicationFactor(0,1),1.2)
        assert_equals(self.getSplineApplicationFactor(0,2),1.5)
        assert_equals(self.getSplineApplicationFactor(0,3),1.8)
        assert_equals(self.getSplineApplicationFactor(2,3),2.8)
    #end def
    
    def test_LifeFactor(self):
        """Test the life factor look up
        
        """
        assert_equals(self.getLifeFactor(1,reversible=False),1.8)
        assert_equals(self.getLifeFactor(1,reversible=True),1.8)
        assert_equals(self.getLifeFactor(9999,reversible=True),1.0)
        assert_equals(self.getLifeFactor(10001,reversible=True),0.4)
    #end def
    
    def test_MaxShaftStress(self):
        """Test the maximum shaft stress calculation
        
        """
        assert_equals(self.maximumShaftStress(2.,3.,5.),1.2)
    #end def
    
    def test_AllowableShearStress(self):
        """Check that the allowable shear stress calc works
        
        """
        assert_almost_equals(\
            self.getAllowableShearStressByHardness(180.,hType='Brinell')/1.0e8,\
            1.37895,places=4)
        assert_almost_equals(\
            self.getAllowableShearStressByHardness(245.,hType='Brinell')/1.0e8,\
            2.06843,places=4)
        assert_almost_equals(\
            self.getAllowableShearStressByHardness(326.,hType='Brinell')/1.0e8,\
            2.7579,places=4)
        assert_almost_equals(\
            self.getAllowableShearStressByHardness(35.,\
                                                   hType='Rockwell C')/1.0e8,\
            2.757901,places=4)
        assert_almost_equals(\
            self.getAllowableShearStressByHardness(44.,\
                                                   hType='Rockwell C')/1.0e8,\
            3.10264,places=4)
        assert_almost_equals(\
            self.getAllowableShearStressByHardness(50.,\
                                                   hType='Rockwell C')/1.0e8,\
            2.757901,places=4)
        assert_almost_equals(\
            self.getAllowableShearStressByHardness(60.,\
                                                   hType='Rockwell C')/1.0e8,\
            3.44738,places=4)
    #end def
    
    def test_ShaftSafetyFactor(self):
        """Check the shaft safety factor calculation
        
        """
        assert_equals(self.getShaftSafetyFactor(2.,3.),1.5)
    #end def
    
    def test_TeethShearStress(self):
        """Check the teeth shear stress calculation
        
        """
        assert_almost_equals(self.getTeethShearStress(2.,3.,5.,7.,11.,13),\
                             0.004795205,\
                             places=4)
    #end def
    
    def test_LoadDistFactorSpline(self):
        """Test the look up of Km
        
        """
        relMis = [0.0005,0.0015,0.003,0.006]
        fE = [0.006,0.018,0.03,0.06]
        lookUp = [[1.,1.,1.,1.5],[1.,1.,1.5,2.],[1.,1.5,2.,2.5],[1.5,2.,2.5,3.]]
        for i in range(len(relMis)):
            for j in range(len(fE)):
                assert_equals(self.getLoadDistributionFactorSpline(relMis[i],\
                              fE[j]),lookUp[i][j])
            #end for
        #end for
    #end def
    
    def test_CompressiveStress(self):
        """Test the compressive stress calculation
        
        """
        assert_almost_equals(self.getCompressiveStress(2.,3.,5.,7.,11.,13),\
                             0.002397602,\
                             places=5)
    #end def
    
    def test_AllowableCompressiveStress(self):
        """Test the lookup of allowable compressive stress
        
        """
        assert_almost_equals(\
            self.getAllowableCompressiveStressForSplines(180.,\
                                               hType='Brinell',\
                                               toothEnd='Straight')/1.0e7,
                                               1.0342,\
                                               places=3)
        assert_almost_equals(\
            self.getAllowableCompressiveStressForSplines(180.,\
                                               hType='Brinell',\
                                               toothEnd='Crowned')/1.0e7,
                                               4.1369,\
                                               places=3)
        assert_almost_equals(\
            self.getAllowableCompressiveStressForSplines(60.,\
                                               hType='Rockwell C',\
                                               toothEnd='Straight')/1.0e7,
                                               3.4474,\
                                               places=3)
        assert_almost_equals(\
            self.getAllowableCompressiveStressForSplines(60.,\
                                               hType='Rockwell C',\
                                               toothEnd='Crowned')/1.0e8,
                                               1.378953593947702,\
                                               places=3)
    #end def
    
    def test_SplineWearLifeFactor(self):
        """Test the lookup of the life wear factor
        
        """
        testList = [[5.0e3,4.],\
                    [5.0e4,2.8],\
                    [5.0e5,2.],\
                    [5.0e6,1.4],\
                    [5.0e7,1.],\
                    [5.0e8,0.7],\
                    [5.0e9,0.5]]
        for t in testList:
            assert_equals(self.getSplineWearLifeFactor(t[0]),t[1])
        #end for
    #end def
    
    def test_AllowableCompressStress(self):
        """Test the allowable compressive stress calculation
        
        """
        assert_almost_equals(\
            self.getAllowableCompressiveStress(2.,3.,5.,flexible=True),\
            1.2, places=1)
        assert_almost_equals(\
            self.getAllowableCompressiveStress(2.,3.,5.,flexible=False),\
            0.133333, places=4)
    #end def
    
    def test_BurstingRadialStress(self):
        """Test the bursting radial stress
        
        """
        assert_almost_equals(\
            self.getBurstingRadialStress(2.,3.,5.,7.,11.)*1.0e5,\
                             -23.5709,\
                             places=4)
    #end def
    
    def test_BurstingCentrifugalStress(self):
        """Test the bursting centrifugal stress
        
        """
        assert_almost_equals(\
            self.getBurstingCentrifugalStress(2,3.,5.)/1.0e3,\
            1.0122969,\
            places=4)
    #end def
    
    def test_BurstingTensileStress(self):
        """Test the bursting tensile stress calculation
        
        """
        assert_almost_equals(\
            self.getBurstingTensileStress(2.,3.,5.,7.),\
            0.025396825,\
            places=4)
    #end def
    
    def test_BurstingTotalStress(self):
        """Test the total stress calculation
        
        """
        assert_equals(self.getTotalBurstingStress(2.,3.,5.,7.,11.),103.)
    #end def
    
    def test_AllowableBurstingStress(self):
        """Test the look up of the bursting stress
        
        """
        testList = [[180,'Brinell',1.5168466],\
                    [240,'Brinell',2.2063223],\
                    [340,'Brinell',3.1026408],\
                    [35,'Rockwell C',3.1026408],\
                    [45,'Rockwell C',3.1026408],\
                    [60,'Rockwell C',3.7921165],\
                    [49,'Rockwell C',3.4473786]]
        for tl in testList:
            assert_almost_equals(\
                self.getAllowableBurstingStressByHardness(\
                                    tl[0],hType=tl[1])/1.0e8,\
                                 tl[2],\
                                 places=4)
        #end for
    #end def
    
    def test_BurstingSafetyFactor(self):
        """Test the bursting safety factor
        
        """
        assert_equals(self.getBurstingSafetyFactor(2.,3.,5.),7.5)
    #end def
    
#end class

class Test_DudleyMethodSpline(DudleyMethodSpline):
    """Test version of the Dudley Method Spline
    
    """
    
    def __init__(self):
        """Blank out the inputs for ease of use
        
        """
        super(Test_DudleyMethodSpline,self).\
            __init__(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
                                 autoCalc=False)
    #end def
    
#end class

class UnrecognisedHardnessTypeException(Exception):
    """Exception thrown when the provided hardness type is not recognised
    
    """
#end class

class UnrecognisedToothEndException(Exception):
    """Exception thrown when the provided hardness type is not recognised
    
    """
#end class
    
import wx
import GUIComponents.Panels as Panels
from GUIComponents.Base import *

class DMSViewer(wx.Frame):
    """Simple viewer GUI for use of the Dudley Method Spline Calculation
    
    """
    
    def __init__(self):
        """Constructor
        
        """
        super(DMSViewer,self).__init__(None,\
                                         -1,\
                                         'Spline Calculation - '+\
                                         'Dudley Method',\
                                         size=wx.Size(800,1200))
        self._parent = None
        self.SetBackgroundColour(wx.Colour(234,232,232))
        self.SetForegroundColour(wx.Colour(48,61,67))
        
        self._splitter = Panels.CoxSplitterWindow(self)
        self._inputForm = DMSEntryPanel(self._splitter)
        self._outputText = Panels.DataProcessingPanel(self._splitter,\
                                                      wx.NewId())
        self._splitter.SplitHorizontally(self._inputForm,self._outputText,720)
        self._inputForm._mainSizer.Layout()
        self._outputText._mainSizer.Layout()
        self.Layout()
    #end def
    
#end class

class DMSEntryPanel(Panels.BasePanel):
    """Panel for the entry of data for the Dudley Method to calculate spline \
    strength
    
    """
    
    def __init__(self,parent):
        """Setup the panel
        
        """
        super(DMSEntryPanel,self).__init__(parent,\
                         wx.NewId(),\
                         'Spline Stress Calculation - Dudley Method')
    #end def
    
    def contentSetup(self):
        """Set up the content of the panel
        
        """
        preamble = 'This application allows the calculation of spline strength'
        preamble += ' using the Dudley Method. This is a basic calculation,'
        preamble += 'but is the industry standard'
        self.addPreText(preamble)
        self._geomTitle = FormLabel(self,\
                                    self._mainSizer,\
                                    'Spline Geometry')
        self._geomTitle.addToSizer()
        self._geomEntry = FormNumericTextEntry(self,\
                ['Bore Diameter, Dh [m]',\
                 'Outside Diameter of Shaft, Doi [m]',\
                 'Pitch Diameter, D [m]',\
                 'Root Diameter of the Shaft, Dre [m]',\
                 'Inner Diameter of the Hub Teeth, [m]',\
                 'Outside Diameter of the Shaft Teeth, Dri [m]',\
                 'Relative Misalignment of Shaft/Hub (misalignment/pitch), []',\
                 'Length of the teeth contact, F [m]',\
                 'Tooth chordal thickness, tC [m]',\
                 'Number of teeth, z []',\
                 'Pressure angle, phi [deg]',\
                 'Hub wall thickness, tW [m]'\
                 ],\
                 self._mainSizer,\
                 alignment=wx.LEFT,\
                 allowNeg=False)
        self._geomEntry.addToSizer()
        self._separators = []
        self._separators.append(FormHorizontalLine(self,\
                                                   self._mainSizer))
        self._separators[-1].addToSizer()
        self._torqTitle = FormLabel(self,\
                                    self._mainSizer,\
                                    'Mechanical Load')
        self._torqTitle.addToSizer()
        self._torqEntry = FormNumericTextEntry(self,\
                                         ['Torque, T [Nm]',\
                                          'Shaft Speed, n [rev/min]'],\
                                         self._mainSizer,\
                                         alignment=wx.LEFT,\
                                         allowNeg=False)
        self._torqEntry.addToSizer()
        self._separators.append(FormHorizontalLine(self,\
                                                   self._mainSizer))
        self._separators[-1].addToSizer()
        self._factDataTitle = FormLabel(self,\
                                        self._mainSizer,\
                                        'General Data')
        self._factDataTitle.addToSizer()
        self._hardness = ['Brinell 160-200',\
                           'Brinell 230-260',\
                           'Brinell 302-351',\
                           'Rockwell C 33-38',\
                           'Rockwell C 42-46',\
                           'Rockwell C 48-53',\
                           'Rockwell C 58-63']
        self._powerSources = ['Uniform (turbine,motor)',\
                              'Light shock (hydraulic motor)',\
                              'Medium shock (internal combustion engine)']
        self._loadTypes = ['Uniform (generator,fan)',\
                           'Light Shock (oscillating, pump,etc.)',\
                           'Intermittent Shock(actuating pumps, etc)',\
                           'Heavy Shock,(punches, shears, etc)']
        self._torqCycs = ['< 1E3','< 1E4','< 1E5','< 1E6','< 1E7']
        self._revs = ['< 1E4','< 1E5','< 1E6','< 1E7','< 1E8','< 1E9','< 1E10']
        self._rot = ['Unidirectional','Fully-reversed']
        self._toothProf = ['Straight','Crowned']
        self._factDataEntry = FormSelection(self,\
                                            ['Material Hardness',\
                                             'Power Source',\
                                             'Load Type',\
                                             'No. of Torque Cycles',\
                                             'No. of Revolutions',\
                                             'Rotation Direction',\
                                             'Tooth profile',\
                                             'Flexible?'],\
                                             [self._hardness,\
                                              self._powerSources,\
                                              self._loadTypes,\
                                              self._torqCycs,\
                                              self._revs,\
                                              self._rot,\
                                              self._toothProf,\
                                              ['Yes','No']],\
                                            self._mainSizer,\
                                            defaultChoices = \
                        ['Rockwell C 58-63',\
                         'Medium shock (internal combustion engine)',\
                         'Uniform (generator,fan)',\
                         '< 1E4',\
                         '< 1E10',\
                         'Unidirectional',\
                         'Straight',\
                         'Yes'],\
                                            alignment = wx.LEFT)
        self._factDataEntry.addToSizer()
        self._separators.append(FormHorizontalLine(self,\
                                                   self._mainSizer))
        self._separators[-1].addToSizer()
        self._tailButtons = FormEndButtons(self,\
                                           'Calculate',\
                                           self.calculateDudley,\
                                           self._tailSizer)
        self._tailButtons.addToSizer()
    #end def
    
    def calculateDudley(self,item):
        """Use the entry data to calculate the spline stresses and safety \
        factors
        
        :param item: Item that launched the method
        :type item: wx.CommandButton
        
        """
        geomData = self._geomEntry.getValues()
        torqData = self._torqEntry.getValues()
        factData = self._factDataEntry.getValues()
        hardnessReference = [['Brinell',180],\
                             ['Brinell',240],\
                             ['Brinell',320],\
                             ['Rockwell C',35],\
                             ['Rockwell C',44],\
                             ['Rockwell C',50],\
                             ['Rockwell C',60]]
        
        hardness = \
            hardnessReference[self._hardness.index(\
                                            factData['Material Hardness'])]
        dms = DudleyMethodSpline(\
                torqData['Torque, T [Nm]'],\
                geomData['Root Diameter of the Shaft, Dre [m]'],\
                self._powerSources.index(factData['Power Source']),\
                self._loadTypes.index(factData['Load Type']),\
                torqData['Shaft Speed, n [rev/min]'],\
                10**float(factData['No. of Torque Cycles'].split('E')[1])-1,\
                10**float(factData['No. of Revolutions'].split('E')[1])-1,\
                hardness[1],\
                geomData['Pitch Diameter, D [m]'],\
                geomData['Number of teeth, z []'],\
                geomData['Length of the teeth contact, F [m]'],\
                geomData['Tooth chordal thickness, tC [m]'],\
                geomData[\
                'Relative Misalignment of Shaft/Hub (misalignment/pitch), []'],\
                (geomData['Outside Diameter of the Shaft Teeth, Dri [m]']-\
                geomData['Inner Diameter of the Hub Teeth, [m]'])/2.0,\
                np.deg2rad(geomData['Pressure angle, phi [deg]']),\
                geomData['Hub wall thickness, tW [m]'],\
                geomData['Length of the teeth contact, F [m]'],\
                geomData['Outside Diameter of Shaft, Doi [m]'],\
                geomData['Outside Diameter of the Shaft Teeth, Dri [m]'],\
                dH=geomData['Bore Diameter, Dh [m]'],\
                hType=hardness[0],\
                reversible=(factData['Rotation Direction'] \
                                            == 'Fully-reversed'),\
                toothEnd=factData['Tooth profile'],\
                flexible=(factData['Flexible?']=='Yes'))
        self._parent._parent._outputText.printToLog(str(dms))
    #end def
    
#end class
