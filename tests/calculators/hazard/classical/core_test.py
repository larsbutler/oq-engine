# Copyright (c) 2010-2012, GEM Foundation.
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.


import getpass
import unittest

import kombu
import numpy

from nose.plugins.attrib import attr

from openquake.calculators.hazard import general
from openquake.calculators.hazard.classical import core
from openquake.db import models
from openquake.utils import stats
from tests.utils import helpers


class ClassicalHazardCalculatorTestCase(unittest.TestCase):
    """
    Tests for the main methods of the classical hazard calculator.
    """

    def setUp(self):
        cfg = helpers.demo_file('simple_fault_demo_hazard/job.ini')
        self.job = helpers.get_hazard_job(cfg, username=getpass.getuser())
        self.calc = core.ClassicalHazardCalculator(self.job)

    def test_pre_execute(self):
        # Most of the pre-execute functionality is implement in other methods.
        # For this test, just make sure each method gets called.
        base_path = ('openquake.calculators.hazard.classical.core'
                     '.ClassicalHazardCalculator')
        init_src_patch = helpers.patch(
            '%s.%s' % (base_path, 'initialize_sources'))
        init_sm_patch = helpers.patch(
            '%s.%s' % (base_path, 'initialize_site_model'))
        init_rlz_patch = helpers.patch(
            '%s.%s' % (base_path, 'initialize_realizations'))
        patches = (init_src_patch, init_sm_patch, init_rlz_patch)

        mocks = [p.start() for p in patches]

        self.calc.pre_execute()

        for i, m in enumerate(mocks):
            self.assertEqual(1, m.call_count)
            m.stop()
            patches[i].stop()

    def test_initalize_sources(self):
        self.calc.initialize_sources()

        # The source model logic tree for this configuration has only 1 source
        # model:
        [source] = models.inputs4hcalc(
            self.job.hazard_calculation.id, input_type='source')

        parsed_sources = models.ParsedSource.objects.filter(input=source)
        # This source model contains 118 sources:
        self.assertEqual(118, len(parsed_sources))

        # Finally, check the Src2ltsrc linkage:
        [smlt] = models.inputs4hcalc(
            self.job.hazard_calculation.id, input_type='lt_source')
        [src2ltsrc] = models.Src2ltsrc.objects.filter(
            hzrd_src=source, lt_src=smlt)
        # Make sure the `filename` is exactly as it apprears in the logic tree.
        # This is import for the logic tree processing we need to do later on.
        self.assertEqual('dissFaultModel.xml', src2ltsrc.filename)

    def test_initialize_site_model(self):
        # we need a slightly different config file for this test
        cfg = helpers.demo_file(
            'simple_fault_demo_hazard/job_with_site_model.ini')
        self.job = helpers.get_hazard_job(cfg)
        self.calc = core.ClassicalHazardCalculator(self.job)

        self.calc.initialize_site_model()
        # If the site model isn't valid for the calculation geometry, a
        # `RuntimeError` should be raised here

        # Okay, it's all good. Now check the count of the site model records.
        [site_model_inp] = models.inputs4hcalc(
            self.job.hazard_calculation.id, input_type='site_model')
        sm_nodes = models.SiteModel.objects.filter(input=site_model_inp)

        self.assertEqual(2601, len(sm_nodes))

        num_pts_to_compute = len(
            self.job.hazard_calculation.points_to_compute())

        [site_data] = models.SiteData.objects.filter(
            hazard_calculation=self.job.hazard_calculation.id)

        # The site model is good. Now test that `site_data` was computed.
        # For now, just test the lengths of the site data collections:
        self.assertEqual(num_pts_to_compute, len(site_data.lons))
        self.assertEqual(num_pts_to_compute, len(site_data.lats))
        self.assertEqual(num_pts_to_compute, len(site_data.vs30s))
        self.assertEqual(num_pts_to_compute, len(site_data.vs30_measured))
        self.assertEqual(num_pts_to_compute, len(site_data.z1pt0s))
        self.assertEqual(num_pts_to_compute, len(site_data.z2pt5s))

    def test_initialize_site_model_no_site_model(self):
        patch_path = 'openquake.calculators.hazard.general.store_site_model'
        with helpers.patch(patch_path) as store_sm_patch:
            self.calc.initialize_site_model()
            # We should never try to store a site model in this case.
            self.assertEqual(0, store_sm_patch.call_count)

    def _check_logic_tree_realization_source_progress(self, ltr):
        # Since the logic for his sample calculation only contains a single
        # source model, both samples will have the number of
        # source_progress records (that is, 1 record per source).
        src_prog = models.SourceProgress.objects.filter(
            lt_realization=ltr.id)
        self.assertEqual(118, len(src_prog))
        self.assertFalse(any([x.is_complete for x in src_prog]))

        # Check that hazard curve progress records were properly
        # initialized:
        [hc_prog_pga] = models.HazardCurveProgress.objects.filter(
            lt_realization=ltr.id, imt="PGA")
        self.assertEqual((120, 19), hc_prog_pga.result_matrix.shape)
        self.assertTrue((hc_prog_pga.result_matrix == 0).all())

        [hc_prog_sa] = models.HazardCurveProgress.objects.filter(
            lt_realization=ltr.id, imt="SA(0.025)")
        self.assertEqual((120, 19), hc_prog_sa.result_matrix.shape)
        self.assertTrue((hc_prog_sa.result_matrix == 0).all())

    def test_initialize_realizations_montecarlo(self):
        # We need initalize sources first (read logic trees, parse sources,
        # etc.)
        self.calc.initialize_sources()

        # No realizations yet:
        ltrs = models.LtRealization.objects.filter(
            hazard_calculation=self.job.hazard_calculation.id)
        self.assertEqual(0, len(ltrs))

        self.calc.initialize_realizations(
            rlz_callbacks=[self.calc.initialize_hazard_curve_progress])

        # We expect 2 logic tree realizations
        ltr1, ltr2 = models.LtRealization.objects.filter(
            hazard_calculation=self.job.hazard_calculation.id).order_by("id")

        # Check each ltr contents, just to be thorough.
        self.assertEqual(0, ltr1.ordinal)
        self.assertEqual(23, ltr1.seed)
        self.assertFalse(ltr1.is_complete)
        self.assertEqual(['b1'], ltr1.sm_lt_path)
        self.assertEqual(['b1'], ltr1.gsim_lt_path)
        self.assertEqual(118, ltr1.total_sources)
        self.assertEqual(0, ltr1.completed_sources)

        self.assertEqual(1, ltr2.ordinal)
        self.assertEqual(1685488378, ltr2.seed)
        self.assertFalse(ltr2.is_complete)
        self.assertEqual(['b1'], ltr2.sm_lt_path)
        self.assertEqual(['b1'], ltr2.gsim_lt_path)
        self.assertEqual(118, ltr2.total_sources)
        self.assertEqual(0, ltr2.completed_sources)

        for ltr in (ltr1, ltr2):
            # Now check that we have source_progress records for each
            # realization.
            self._check_logic_tree_realization_source_progress(ltr)

    def test_initialize_pr_data(self):
        # The total/done counters for progress reporting are initialized
        # correctly.
        self.calc.initialize_sources()
        self.calc.initialize_realizations(
            rlz_callbacks=[self.calc.initialize_hazard_curve_progress])
        ltr1, ltr2 = models.LtRealization.objects.filter(
            hazard_calculation=self.job.hazard_calculation.id).order_by("id")

        ltr1.completed_sources = 11
        ltr1.save()

        self.calc.initialize_pr_data()

        total = stats.pk_get(self.calc.job.id, "nhzrd_total")
        self.assertEqual(ltr1.total_sources + ltr2.total_sources, total)
        done = stats.pk_get(self.calc.job.id, "nhzrd_done")
        self.assertEqual(ltr1.completed_sources + ltr2.completed_sources, done)

    def test_initialize_realizations_enumeration(self):
        self.calc.initialize_sources()
        # enumeration is triggered by zero value used as number of realizations
        self.calc.job.hazard_calculation.number_of_logic_tree_samples = 0
        self.calc.initialize_realizations(
            rlz_callbacks=[self.calc.initialize_hazard_curve_progress])

        [ltr] = models.LtRealization.objects.filter(
            hazard_calculation=self.job.hazard_calculation.id)

        # Check each ltr contents, just to be thorough.
        self.assertEqual(0, ltr.ordinal)
        self.assertEqual(None, ltr.seed)
        self.assertFalse(ltr.is_complete)
        self.assertEqual(['b1'], ltr.sm_lt_path)
        self.assertEqual(['b1'], ltr.gsim_lt_path)
        self.assertEqual(118, ltr.total_sources)
        self.assertEqual(0, ltr.completed_sources)

        self._check_logic_tree_realization_source_progress(ltr)

    @attr('slow')
    def test_complete_calculation_workflow(self):
        # Test the calculation workflow, from pre_execute through clean_up
        # TODO: `post_process` is skipped for the moment until this
        # functionality is available
        hc = self.job.hazard_calculation

        self.calc.pre_execute()

        # Update job status to move on to the execution phase.
        self.job.is_running = True

        self.job.status = 'executing'
        self.job.save()
        self.calc.execute()

        self.job.status = 'post_executing'
        self.job.save()
        self.calc.post_execute()

        lt_rlzs = models.LtRealization.objects.filter(
            hazard_calculation=self.job.hazard_calculation.id)

        self.assertEqual(2, len(lt_rlzs))

        # Now we test that the htemp results were copied to the final location
        # in `hzrdr.hazard_curve` and `hzrdr.hazard_curve_data`.
        for rlz in lt_rlzs:
            self.assertEqual(rlz.total_sources, rlz.completed_sources)
            self.assertTrue(rlz.is_complete)

            # get hazard curves for this realization
            pga_curves = models.HazardCurve.objects.get(
                lt_realization=rlz.id, imt='PGA')
            sa_curves = models.HazardCurve.objects.get(
                lt_realization=rlz.id, imt='SA', sa_period=0.025)

            # In this calculation, we have 120 sites of interest.
            # We should have exactly that many curves per realization
            # per IMT.
            pga_curve_data = models.HazardCurveData.objects.filter(
                hazard_curve=pga_curves.id)
            self.assertEqual(120, len(pga_curve_data))
            sa_curve_data = models.HazardCurveData.objects.filter(
                hazard_curve=sa_curves.id)
            self.assertEqual(120, len(sa_curve_data))

        self.job.status = 'post_processing'
        self.job.save()
        self.calc.post_process()

        # Now test for the existence (and quantity) of mean curves:
        pga_mean_curve_set = models.HazardCurve.objects.get(
            output__oq_job=self.job,
            statistics='mean',
            imt='PGA')
        pga_mean_curves = models.HazardCurveData.objects.filter(
            hazard_curve=pga_mean_curve_set)
        # 1 mean curve per location for this imt
        self.assertEqual(120, pga_mean_curves.count())

        sa_mean_curve_set = models.HazardCurve.objects.get(
            output__oq_job=self.job,
            statistics='mean',
            imt='SA',
            sa_period=0.025)
        sa_mean_curves = models.HazardCurveData.objects.filter(
            hazard_curve=sa_mean_curve_set)
        # 1 mean curve per location for this imt
        self.assertEqual(120, sa_mean_curves.count())

        # Test `clean_up`
        self.job.status = 'clean_up'
        self.job.save()
        self.calc.clean_up()

        # last thing, make sure that `clean_up` cleaned up the htemp tables
        hcp = models.HazardCurveProgress.objects.filter(
            lt_realization__hazard_calculation=hc.id)
        self.assertEqual(0, len(hcp))

        sp = models.SourceProgress.objects.filter(
            lt_realization__hazard_calculation=hc.id)
        self.assertEqual(0, len(sp))

        sd = models.SiteData.objects.filter(hazard_calculation=hc.id)
        self.assertEqual(0, len(sd))

    def test_hazard_curves_task(self):
        # Test the `hazard_curves` task, but execute it as a normal function
        # (for purposes of test coverage).
        hc = self.job.hazard_calculation

        self.calc.pre_execute()

        # Update job status to move on to the execution phase.
        self.job.is_running = True

        self.job.status = 'executing'
        self.job.save()

        src_prog = models.SourceProgress.objects.filter(
            is_complete=False,
            lt_realization__hazard_calculation=hc).latest('id')

        src_id = src_prog.parsed_source.id
        lt_rlz = src_prog.lt_realization

        exchange, conn_args = general.exchange_and_conn_args()

        routing_key = general.ROUTING_KEY_FMT % dict(job_id=self.job.id)
        task_signal_queue = kombu.Queue(
            'htasks.job.%s' % self.job.id, exchange=exchange,
            routing_key=routing_key, durable=False, auto_delete=True)

        def test_callback(body, message):
            self.assertEqual(dict(job_id=self.job.id, num_sources=1),
                             body)
            message.ack()

        with kombu.BrokerConnection(**conn_args) as conn:
            task_signal_queue(conn.channel()).declare()
            with conn.Consumer(task_signal_queue, callbacks=[test_callback]):
                # call the task as a normal function
                core.hazard_curves(self.job.id, [src_id], lt_rlz.id)
                # wait for the completion signal
                conn.drain_events()

        # refresh the source_progress record and make sure it is marked as
        # complete
        src_prog = models.SourceProgress.objects.get(id=src_prog.id)
        self.assertTrue(src_prog.is_complete)
        # We'll leave more detail testing of results to a QA test (which will
        # take much more time to execute).


class HelpersTestCase(unittest.TestCase):
    """
    Tests for helper functions in the classical hazard calculator core module.
    """

    def test_update_result_matrix_with_scalars(self):
        init = 0.0
        result = core.update_result_matrix(init, 0.2)
        # The first time we apply this formula on a 0.0 value,
        # result is equal to the first new value we apply.
        self.assertAlmostEqual(0.2, result)

        result = core.update_result_matrix(result, 0.3)
        self.assertAlmostEqual(0.44, result)

    def test_update_result_matrix_numpy_arrays(self):
        init = numpy.zeros((4, 4))
        first = numpy.array([0.2] * 16).reshape((4, 4))

        result = core.update_result_matrix(init, first)
        numpy.testing.assert_allclose(first, result)

        second = numpy.array([0.3] * 16).reshape((4, 4))
        result = core.update_result_matrix(result, second)

        expected = numpy.array([0.44] * 16).reshape((4, 4))
        numpy.testing.assert_allclose(expected, result)

    def test_compute_mean_curve(self):
        curves = [
            [1.0, 0.85, 0.67, 0.3],
            [0.87, 0.76, 0.59, 0.21],
            [0.62, 0.41, 0.37, 0.0],
        ]

        expected_mean_curve = numpy.array([0.83, 0.67333333, 0.54333333, 0.17])
        numpy.testing.assert_allclose(
            expected_mean_curve, core.compute_mean_curve(curves))

    def test_compute_mean_curve_weighted(self):
        curves = [
            [1.0, 0.85, 0.67, 0.3],
            [0.87, 0.76, 0.59, 0.21],
            [0.62, 0.41, 0.37, 0.0],
        ]
        weights = [0.5, 0.3, 0.2]

        expected_mean_curve = numpy.array([0.885, 0.735, 0.586, 0.213])
        numpy.testing.assert_allclose(
            expected_mean_curve,
            core.compute_mean_curve(curves, weights=weights))

    def test_compute_mean_curve_weights_None(self):
        # If all weight values are None, ignore the weights altogether
        curves = [
            [1.0, 0.85, 0.67, 0.3],
            [0.87, 0.76, 0.59, 0.21],
            [0.62, 0.41, 0.37, 0.0],
        ]
        weights = [None, None, None]

        expected_mean_curve = numpy.array([0.83, 0.67333333, 0.54333333, 0.17])
        numpy.testing.assert_allclose(
            expected_mean_curve,
            core.compute_mean_curve(curves, weights=weights))

    def test_compute_mean_curve_invalid_weights(self):
        curves = [
            [1.0, 0.85, 0.67, 0.3],
            [0.87, 0.76, 0.59, 0.21],
            [0.62, 0.41, 0.37, 0.0],
        ]
        weights = [0.6, None, 0.4]

        self.assertRaises(ValueError, core.compute_mean_curve, curves, weights)

    def test_compute_quantile_curve(self):
        expected_curve = numpy.array([
            9.9178000e-01, 9.8892000e-01, 9.6903000e-01, 9.4030000e-01,
            8.8405000e-01, 7.8782000e-01, 6.4897250e-01, 4.8284250e-01,
            3.4531500e-01, 3.2337000e-01, 1.8880500e-01, 9.5574000e-02,
            4.3707250e-02, 1.9643000e-02, 8.1923000e-03, 2.9157000e-03,
            7.9955000e-04, 1.5233000e-04, 1.5582000e-05])

        quantile = 0.75

        curves = [
            [9.8161000e-01, 9.7837000e-01, 9.5579000e-01, 9.2555000e-01,
             8.7052000e-01, 7.8214000e-01, 6.5708000e-01, 5.0526000e-01,
             3.7044000e-01, 3.4740000e-01, 2.0502000e-01, 1.0506000e-01,
             4.6531000e-02, 1.7548000e-02, 5.4791000e-03, 1.3377000e-03,
             2.2489000e-04, 2.2345000e-05, 4.2696000e-07],
            [9.7309000e-01, 9.6857000e-01, 9.3853000e-01, 9.0089000e-01,
             8.3673000e-01, 7.4057000e-01, 6.1272000e-01, 4.6467000e-01,
             3.3694000e-01, 3.1536000e-01, 1.8340000e-01, 9.2412000e-02,
             4.0202000e-02, 1.4900000e-02, 4.5924000e-03, 1.1126000e-03,
             1.8647000e-04, 1.8882000e-05, 4.7123000e-07],
            [9.9178000e-01, 9.8892000e-01, 9.6903000e-01, 9.4030000e-01,
             8.8405000e-01, 7.8782000e-01, 6.4627000e-01, 4.7537000e-01,
             3.3168000e-01, 3.0827000e-01, 1.7279000e-01, 8.8360000e-02,
             4.2766000e-02, 1.9643000e-02, 8.1923000e-03, 2.9157000e-03,
             7.9955000e-04, 1.5233000e-04, 1.5582000e-05],
            [9.8885000e-01, 9.8505000e-01, 9.5972000e-01, 9.2494000e-01,
             8.6030000e-01, 7.5574000e-01, 6.1009000e-01, 4.4217000e-01,
             3.0543000e-01, 2.8345000e-01, 1.5760000e-01, 8.0225000e-02,
             3.8681000e-02, 1.7637000e-02, 7.2685000e-03, 2.5474000e-03,
             6.8347000e-04, 1.2596000e-04, 1.2853000e-05],
            [9.9178000e-01, 9.8892000e-01, 9.6903000e-01, 9.4030000e-01,
             8.8405000e-01, 7.8782000e-01, 6.4627000e-01, 4.7537000e-01,
             3.3168000e-01, 3.0827000e-01, 1.7279000e-01, 8.8360000e-02,
             4.2766000e-02, 1.9643000e-02, 8.1923000e-03, 2.9157000e-03,
             7.9955000e-04, 1.5233000e-04, 1.5582000e-05],
        ]
        actual_curve = core.compute_quantile_curve(curves, quantile)

        # TODO(LB): Check with our hazard experts to see if this is reasonable
        # tolerance. Better yet, get a fresh set of test data. (This test data
        # was just copied verbatim from an existing test. See
        # tests/hazard_test.py.)
        numpy.testing.assert_allclose(expected_curve, actual_curve, atol=0.005)

    def test_compute_weighted_quantile_curve_case1(self):
        expected_curve = numpy.array([0.69909, 0.60859, 0.50328])

        quantile = 0.3

        curves = [
            [9.9996e-01, 9.9962e-01, 9.9674e-01],
            [6.9909e-01, 6.0859e-01, 5.0328e-01],
            [1.0000e+00, 9.9996e-01, 9.9947e-01],
        ]
        weights = [0.5, 0.3, 0.2]

        actual_curve = core.compute_weighted_quantile_curve(
            curves, weights, quantile)

        numpy.testing.assert_allclose(expected_curve, actual_curve)

    def test_compute_weighted_quantile_curve_case2(self):
        expected_curve = numpy.array([0.89556, 0.83045, 0.73646])

        quantile = 0.3

        curves = [
            [9.2439e-01, 8.6700e-01, 7.7785e-01],
            [8.9556e-01, 8.3045e-01, 7.3646e-01],
            [9.1873e-01, 8.6697e-01, 7.8992e-01],
        ]
        weights = [0.2, 0.3, 0.5]

        actual_curve = core.compute_weighted_quantile_curve(
            curves, weights, quantile)

        numpy.testing.assert_allclose(expected_curve, actual_curve)
