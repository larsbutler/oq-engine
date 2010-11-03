package org.gem.calc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.gem.calc.HazardCalculator;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.opensha.commons.calc.magScalingRelations.magScalingRelImpl.WC1994_MagLengthRelationship;
import org.opensha.commons.data.Site;
import org.opensha.commons.geo.BorderType;
import org.opensha.commons.geo.GriddedRegion;
import org.opensha.commons.geo.Location;
import org.opensha.commons.geo.LocationList;
import org.opensha.commons.geo.Region;
import org.opensha.commons.param.DoubleParameter;
import org.opensha.commons.param.event.ParameterChangeWarningListener;
import org.opensha.sha.earthquake.EqkRupForecast;
import org.opensha.sha.earthquake.EqkRupForecastAPI;
import org.opensha.sha.earthquake.ProbEqkSource;
import org.opensha.sha.earthquake.rupForecastImpl.FloatingPoissonFaultSource;
import org.opensha.sha.faultSurface.FaultTrace;
import org.opensha.sha.faultSurface.StirlingGriddedSurface;
import org.opensha.sha.imr.ScalarIntensityMeasureRelationshipAPI;
import org.opensha.sha.imr.attenRelImpl.BA_2008_AttenRel;
import org.opensha.sha.imr.param.IntensityMeasureParams.PGA_Param;
import org.opensha.sha.imr.param.OtherParams.ComponentParam;
import org.opensha.sha.imr.param.OtherParams.SigmaTruncLevelParam;
import org.opensha.sha.imr.param.OtherParams.SigmaTruncTypeParam;
import org.opensha.sha.imr.param.OtherParams.StdDevTypeParam;
import org.opensha.sha.imr.param.SiteParams.Vs30_Param;
import org.opensha.sha.magdist.GutenbergRichterMagFreqDist;
import org.opensha.sha.util.TectonicRegionType;

public class HazardCalculatorTest {

    private static List<Site> siteList;
    private static EqkRupForecastAPI erf;
    private static Map<TectonicRegionType, ScalarIntensityMeasureRelationshipAPI> gmpeMap;
    private static List<Double> imlVals;
    private static double integrationDistance = 200.0;
    private static Random rn = new Random();

    @Before
    public void setUp() {
        setUpSites();
        setUpErf();
        setUpGmpeMap();
        setUpImlValues();
    }

    @After
    public void tearDown() {
        siteList = null;
        erf = null;
        gmpeMap = null;
        imlVals = null;
    }

    /**
     * Test getHazardCurves when a null list of site is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getHazardCurvesNullSiteList() {
        ArrayList<Site> siteList = null;
        HazardCalculator.getHazardCurves(siteList, erf, gmpeMap, imlVals,
                integrationDistance);
    }

    /**
     * Test getHazardCurves when an empty list of site is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getHazardCurvesEmptySiteList() {
        ArrayList<Site> siteList = new ArrayList<Site>();
        HazardCalculator.getHazardCurves(siteList, erf, gmpeMap, imlVals,
                integrationDistance);
    }

    /**
     * Test getHazardCurves when a null earthquake rupture forecast is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getHazardCurvesNullErf() {
        EqkRupForecastAPI erf = null;
        HazardCalculator.getHazardCurves(siteList, erf, gmpeMap, imlVals,
                integrationDistance);
    }

    /**
     * Test getHazardCurves when a null gmpe map is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getHazardCurvesNullGmpeMap() {
        Map<TectonicRegionType, ScalarIntensityMeasureRelationshipAPI> gmpeMap =
                null;
        HazardCalculator.getHazardCurves(siteList, erf, gmpeMap, imlVals,
                integrationDistance);
    }

    /**
     * Test getHazardCurves when an empty gmpe map is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getHazardCurvesEmptyGmpeMap() {
        Map<TectonicRegionType, ScalarIntensityMeasureRelationshipAPI> gmpeMap =
                new HashMap<TectonicRegionType, ScalarIntensityMeasureRelationshipAPI>();
        HazardCalculator.getHazardCurves(siteList, erf, gmpeMap, imlVals,
                integrationDistance);
    }

    /**
     * Test getHazardCurves when a null intensity measure levels array is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getHazardCurvesNullImlLevelsList() {
        List<Double> imlVals = null;
        HazardCalculator.getHazardCurves(siteList, erf, gmpeMap, imlVals,
                integrationDistance);
    }

    /**
     * Test getHazardCurves when an empty intensity measure levels array is
     * passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getHazardCurvesEmptyImlLevelsList() {
        List<Double> imlVals = new ArrayList<Double>();
        HazardCalculator.getHazardCurves(siteList, erf, gmpeMap, imlVals,
                integrationDistance);
    }

    /**
     * Test getGroundMotionFields when a null list of site is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getGroundMotionFieldsNullSiteList() {
        ArrayList<Site> siteList = null;
        HazardCalculator.getGroundMotionFields(siteList, erf, gmpeMap, rn);
    }

    /**
     * Test getGroundMotionFields when an empty list of site is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getGroundMotionFieldsEmptySiteList() {
        ArrayList<Site> siteList = new ArrayList<Site>();
        HazardCalculator.getGroundMotionFields(siteList, erf, gmpeMap, rn);
    }

    /**
     * Test getGroundMotionFields when a null earthquake rupture forecast is
     * passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getGroundMotionFieldsNullErf() {
        EqkRupForecastAPI erf = null;
        HazardCalculator.getGroundMotionFields(siteList, erf, gmpeMap, rn);
    }

    /**
     * Test getGroundMotionFields when a null gmpe map is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getGroundMotionFieldsNullGmpeMap() {
        Map<TectonicRegionType, ScalarIntensityMeasureRelationshipAPI> gmpeMap =
                null;
        HazardCalculator.getGroundMotionFields(siteList, erf, gmpeMap, rn);
    }

    /**
     * Test getGroundMotionFields when an empty gmpe map is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getGroundMotionFieldsEmptyGmpeMap() {
        Map<TectonicRegionType, ScalarIntensityMeasureRelationshipAPI> gmpeMap =
                new HashMap<TectonicRegionType, ScalarIntensityMeasureRelationshipAPI>();
        HazardCalculator.getGroundMotionFields(siteList, erf, gmpeMap, rn);
    }

    /**
     * Test getGroundMotionFields when null random number is passed
     */
    @Test(expected = IllegalArgumentException.class)
    public void getGroundMotionFieldsNullRandomNumberGenerator() {
        Random rn = null;
        HazardCalculator.getGroundMotionFields(siteList, erf, gmpeMap, rn);
    }

    /**
     * Set up list of sites
     */
    private static void setUpSites() {
        siteList = new ArrayList<Site>();
        LocationList border = new LocationList();
        border.add(new Location(35.0, 35.0));
        border.add(new Location(35.0, 38.0));
        border.add(new Location(38.0, 38.0));
        border.add(new Location(38.0, 35.0));
        Region reg = new Region(border, BorderType.MERCATOR_LINEAR);
        double spacing = 0.1;
        GriddedRegion griddedReg = new GriddedRegion(reg, spacing, null);
        for (Location loc : griddedReg.getNodeList()) {
            Site site = new Site(loc);
            site.addParameter(new DoubleParameter(Vs30_Param.NAME, 760.0));
            siteList.add(site);
        }
    }

    /**
     * Set up ERF
     */
    private void setUpErf() {
        erf = new EqkRupForecast() {

            @Override
            public String getName() {
                return new String(
                        "Earthquake rupture forecast for testing pourpose");
            }

            @Override
            public void updateForecast() {
            }

            @Override
            public ArrayList getSourceList() {
                ArrayList<ProbEqkSource> list = new ArrayList<ProbEqkSource>();
                list.add(getFloatingPoissonFaultSource());
                return list;
            }

            @Override
            public ProbEqkSource getSource(int iSource) {
                return getFloatingPoissonFaultSource();
            }

            @Override
            public int getNumSources() {
                return 1;
            }
        };
    }

    /**
     * Set up gmpe map
     */
    private void setUpGmpeMap() {
        gmpeMap =
                new Hashtable<TectonicRegionType, ScalarIntensityMeasureRelationshipAPI>();
        ParameterChangeWarningListener warningListener = null;
        BA_2008_AttenRel imr = new BA_2008_AttenRel(warningListener);
        imr.setParamDefaults();
        imr.getParameter(StdDevTypeParam.NAME).setValue(
                StdDevTypeParam.STD_DEV_TYPE_TOTAL);
        imr.getParameter(SigmaTruncTypeParam.NAME).setValue(
                SigmaTruncTypeParam.SIGMA_TRUNC_TYPE_2SIDED);
        imr.getParameter(SigmaTruncLevelParam.NAME).setValue(3.0);
        imr.setIntensityMeasure(PGA_Param.NAME);
        imr.getParameter(ComponentParam.NAME).setValue(
                ComponentParam.COMPONENT_GMRotI50);
        gmpeMap.put(TectonicRegionType.ACTIVE_SHALLOW, imr);
    }

    /**
     * Set up intensity measure levels
     */
    private void setUpImlValues() {
        imlVals = new ArrayList<Double>();
        imlVals.add(Math.log(0.005));
        imlVals.add(Math.log(0.007));
        imlVals.add(Math.log(0.0098));
        imlVals.add(Math.log(0.0137));
        imlVals.add(Math.log(0.0192));
        imlVals.add(Math.log(0.0269));
        imlVals.add(Math.log(0.0376));
        imlVals.add(Math.log(0.0527));
        imlVals.add(Math.log(0.0738));
        imlVals.add(Math.log(0.103));
        imlVals.add(Math.log(0.145));
        imlVals.add(Math.log(0.203));
        imlVals.add(Math.log(0.284));
        imlVals.add(Math.log(0.397));
        imlVals.add(Math.log(0.556));
        imlVals.add(Math.log(0.778));
        imlVals.add(Math.log(1.09));
    }

    /**
     * Defines fault source (data taken from Turkey model)
     * 
     * @return
     */
    private static FloatingPoissonFaultSource getFloatingPoissonFaultSource() {
        FaultTrace trace = new FaultTrace("trf41");
        trace.add(new Location(37.413314, 36.866757));
        trace.add(new Location(37.033241, 36.640297));
        trace.add(new Location(36.608673, 36.431566));
        trace.add(new Location(36.488077, 36.375783));
        trace.add(new Location(35.677685, 36.271872));
        double occurrenceRate = 0.017;
        double beta = 2.115;
        double mMin = 6.8;
        double mMax = 7.9;
        double mfdBinWidth = 0.1;
        int num = (int) ((mMax - mMin) / mfdBinWidth + 1);
        GutenbergRichterMagFreqDist mfd =
                new GutenbergRichterMagFreqDist(mMin, mMax, num);
        mfd.setAllButTotMoRate(mMin, mMax, occurrenceRate, beta / Math.log(10));
        double dip = 90.0;
        double rake = 0.0;
        double upperSeismogenicDepth = 0.0;
        double lowerSeismogenicDepth = 0.0;
        double gridSpacing = 1.0;
        StirlingGriddedSurface surf =
                new StirlingGriddedSurface(trace, dip, upperSeismogenicDepth,
                        lowerSeismogenicDepth, gridSpacing);
        FloatingPoissonFaultSource src =
                new FloatingPoissonFaultSource(mfd, surf,
                        new WC1994_MagLengthRelationship(), 0.0, 1.5, 1.0,
                        rake, 50.0, mMin, 1, 12.0);
        src.setTectonicRegionType(TectonicRegionType.ACTIVE_SHALLOW);
        return src;
    }
}