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

"""
Core functionality for the classical PSHA hazard calculator.
"""

import re

import nhlib
import nhlib.calc
import nhlib.imt
import numpy

from django.db import transaction
from scipy.stats import mstats

from openquake import logs
from openquake import writer
from openquake.calculators.hazard import general as haz_general
from openquake.db import models
from openquake.input import logictree
from openquake.utils import stats
from openquake.utils import tasks as utils_tasks


# Silencing 'Too many local variables'
# pylint: disable=R0914
@utils_tasks.oqtask
@stats.count_progress('h')
def hazard_curves(job_id, src_ids, lt_rlz_id):
    """
    Celery task for hazard curve calculator.

    Samples logic trees, gathers site parameters, and calls the hazard curve
    calculator.

    Once hazard curve data is computed, result progress updated (within a
    transaction, to prevent race conditions) in the
    `htemp.hazard_curve_progress` table.

    Once all of this work is complete, a signal will be sent via AMQP to let
    the control node know that the work is complete. (If there is any work left
    to be dispatched, this signal will indicate to the control node that more
    work can be enqueued.)

    :param int job_id:
        ID of the currently running job.
    :param src_ids:
        List of ids of parsed source models to take into account.
    :param lt_rlz_id:
        Id of logic tree realization model to calculate for.
    """
    logs.LOG.debug('> starting task: job_id=%s, lt_realization_id=%s'
                   % (job_id, lt_rlz_id))

    hc = models.HazardCalculation.objects.get(oqjob=job_id)

    lt_rlz = models.LtRealization.objects.get(id=lt_rlz_id)
    ltp = logictree.LogicTreeProcessor(hc.id)

    apply_uncertainties = ltp.parse_source_model_logictree_path(
            lt_rlz.sm_lt_path)
    gsims = ltp.parse_gmpe_logictree_path(lt_rlz.gsim_lt_path)

    sources = haz_general.gen_sources(
        src_ids, apply_uncertainties, hc.rupture_mesh_spacing,
        hc.width_of_mfd_bin, hc.area_source_discretization)

    imts = haz_general.im_dict_to_nhlib(hc.intensity_measure_types_and_levels)

    # Now initialize the site collection for use in the calculation.
    # If there is no site model defined, we will use the same reference
    # parameters (defined in the HazardCalculation) for every site.

    # TODO: We could just create the SiteCollection once, pickle it, and store
    # it in the DB (in SiteData). Creating the SiteCollection isn't an
    # expensive operation (at least for small calculations), but this is
    # wasted work.
    logs.LOG.debug('> creating site collection')
    site_coll = haz_general.get_site_collection(hc)
    logs.LOG.debug('< done creating site collection')

    # Prepare args for the calculator.
    calc_kwargs = {'gsims': gsims,
                   'truncation_level': hc.truncation_level,
                   'time_span': hc.investigation_time,
                   'sources': sources,
                   'imts': imts,
                   'sites': site_coll}

    if hc.maximum_distance:
        dist = hc.maximum_distance
        calc_kwargs['source_site_filter'] = (
                nhlib.calc.filters.source_site_distance_filter(dist))
        calc_kwargs['rupture_site_filter'] = (
                nhlib.calc.filters.rupture_site_distance_filter(dist))

    # mapping "imt" to 2d array of hazard curves: first dimension -- sites,
    # second -- IMLs
    logs.LOG.debug('> computing hazard matrices')
    matrices = nhlib.calc.hazard_curve.hazard_curves_poissonian(**calc_kwargs)
    logs.LOG.debug('< done computing hazard matrices')

    logs.LOG.debug('> starting transaction')
    with transaction.commit_on_success():
        logs.LOG.debug('looping over IMTs')

        for imt in hc.intensity_measure_types_and_levels.keys():
            logs.LOG.debug('> updating hazard for IMT=%s' % imt)
            nhlib_imt = haz_general.imt_to_nhlib(imt)
            query = """
            SELECT * FROM htemp.hazard_curve_progress
            WHERE lt_realization_id = %s
            AND imt = %s
            FOR UPDATE"""
            [hc_progress] = models.HazardCurveProgress.objects.raw(
                query, [lt_rlz.id, imt])

            hc_progress.result_matrix = update_result_matrix(
                hc_progress.result_matrix, matrices[nhlib_imt])
            hc_progress.save()

            logs.LOG.debug('< done updating hazard for IMT=%s' % imt)

        # Before the transaction completes:

        # Check here if any of records in source progress model
        # with parsed_source_id from src_ids are marked as complete,
        # and rollback and abort if there is at least one
        src_prog = models.SourceProgress.objects.filter(
            lt_realization=lt_rlz, parsed_source__in=src_ids)

        if any(x.is_complete for x in src_prog):
            msg = (
                'One or more `source_progress` records were marked as '
                'complete. This was unexpected and probably means that the'
                ' calculation workload was not distributed properly.'
            )
            logs.LOG.critical(msg)
            transaction.rollback()
            raise RuntimeError(msg)

        # Mark source_progress records as complete
        src_prog.update(is_complete=True)

        # Update realiation progress,
        # mark realization as complete if it is done
        # First, refresh the logic tree realization record:
        lt_rlz = models.LtRealization.objects.get(id=lt_rlz.id)

        lt_rlz.completed_sources += len(src_ids)
        if lt_rlz.completed_sources == lt_rlz.total_sources:
            lt_rlz.is_complete = True

        lt_rlz.save()

    logs.LOG.debug('< transaction complete')

    # Last thing, signal back the control node to indicate the completion of
    # task. The control node needs this to manage the task distribution and
    # keep track of progress.
    logs.LOG.debug('< task complete, signalling completion')
    haz_general.signal_task_complete(job_id, len(src_ids))


@staticmethod
def classical_task_arg_gen(hc, job, sources_per_task, progress):
    """
    Loop through realizations and sources to generate a sequence of
    task arg tuples. Each tuple of args applies to a single task.

    Yielded results are triples of (job_id, realization_id,
    source_id_list).

    :param hc:
        :class:`openquake.db.models.HazardCalculation` instance.
    :param job:
        :class:`openquake.db.models.OqJob` instance.
    :param int sources_per_task:
        The (max) number of sources to consider for each task.
    :param dict progress:
        A dict containing two integer values: 'total' and 'computed'. The task
        arg generator will update the 'total' count as the generator creates
        arguments.
    """
    realizations = models.LtRealization.objects.filter(
            hazard_calculation=hc, is_complete=False)

    for lt_rlz in realizations:
        source_progress = models.SourceProgress.objects.filter(
                is_complete=False, lt_realization=lt_rlz).order_by('id')
        source_ids = source_progress.values_list('parsed_source_id',
                                                 flat=True)
        progress['total'] += len(source_ids)

        for offset in xrange(0, len(source_ids), sources_per_task):
            task_args = (job.id, source_ids[offset:offset + sources_per_task],
                         lt_rlz.id)
            yield task_args


class ClassicalHazardCalculator(haz_general.BaseHazardCalculatorNext):
    """
    Classical PSHA hazard calculator. Computes hazard curves for a given set of
    points.

    For each realization of the calculation, we randomly sample source models
    and GMPEs (Ground Motion Prediction Equations) from logic trees.
    """

    core_calc_task = hazard_curves
    task_arg_gen = classical_task_arg_gen

    def initialize_hazard_curve_progress(self, lt_rlz):
        """
        As a calculation progresses, workers will periodically update the
        intermediate results. These results will be stored in
        `htemp.hazard_curve_progress` until the calculation is completed.

        Before the core calculation begins, we need to initalize these records,
        one data set per IMT. Each dataset will be stored in the database as a
        pickled 2D numpy array (with number of rows == calculation points of
        interest and number of columns == number of IML values for a given
        IMT).

        We will create 1 `hazard_curve_progress` record per IMT per
        realization.

        :param lt_rlz:
            :class:`openquake.db.models.LtRealization` object to associate
            with these inital hazard curve values.
        """
        hc = self.job.hazard_calculation

        num_points = len(hc.points_to_compute())

        im_data = hc.intensity_measure_types_and_levels
        for imt, imls in im_data.items():
            hc_prog = models.HazardCurveProgress()
            hc_prog.lt_realization = lt_rlz
            hc_prog.imt = imt
            hc_prog.result_matrix = numpy.zeros((num_points, len(imls)))
            hc_prog.save()

    def pre_execute(self):
        """
        Do pre-execution work. At the moment, this work entails: parsing and
        initializing sources, parsing and initializing the site model (if there
        is one), and generating logic tree realizations. (The latter piece
        basically defines the work to be done in the `execute` phase.)
        """

        # Parse logic trees and create source Inputs.
        self.initialize_sources()

        # Deal with the site model and compute site data for the calculation
        # (if a site model was specified, that is).
        self.initialize_site_model()

        # Now bootstrap the logic tree realizations and related data.
        # This defines for us the "work" that needs to be done when we reach
        # the `execute` phase.
        # This will also stub out hazard curve result records. Workers will
        # update these periodically with partial results (partial meaning,
        # result curves for just a subset of the overall sources) when some
        # work is complete.
        self.initialize_realizations(
            rlz_callbacks=[self.initialize_hazard_curve_progress])
        self.initialize_pr_data()

    def post_execute(self):
        """
        Create the final output records for hazard curves. This is done by
        copying the temporary results from `htemp.hazard_curve_progress` to
        `hzrdr.hazard_curve` (for metadata) and `hzrdr.hazard_curve_data` (for
        the actual curve PoE values). Foreign keys are made from
        `hzrdr.hazard_curve` to `hzrdr.lt_realization` (realization information
        is need to export the full hazard curve results).
        """
        hc = self.job.hazard_calculation
        im = hc.intensity_measure_types_and_levels
        points = hc.points_to_compute()

        realizations = models.LtRealization.objects.filter(
            hazard_calculation=hc.id)

        for rlz in realizations:
            # create a new `HazardCurve` 'container' record for each
            # realization for each intensity measure type
            for imt, imls in im.items():
                sa_period = None
                sa_damping = None
                if 'SA' in imt:
                    match = re.match(r'^SA\(([^)]+?)\)$', imt)
                    sa_period = float(match.group(1))
                    sa_damping = haz_general.DEFAULT_SA_DAMPING
                    hc_im_type = 'SA'  # don't include the period
                else:
                    hc_im_type = imt

                hco = models.Output(
                    owner=hc.owner,
                    oq_job=self.job,
                    display_name="hc-rlz-%s" % rlz.id,
                    output_type='hazard_curve',
                )
                hco.save()

                haz_curve = models.HazardCurve(
                    output=hco,
                    lt_realization=rlz,
                    investigation_time=hc.investigation_time,
                    imt=hc_im_type,
                    imls=imls,
                    sa_period=sa_period,
                    sa_damping=sa_damping,
                )
                haz_curve.save()

                [hc_progress] = models.HazardCurveProgress.objects.filter(
                    lt_realization=rlz.id, imt=imt)

                with transaction.commit_on_success(using='reslt_writer'):
                    hc_data_inserter = writer.BulkInserter(
                        models.HazardCurveData)
                    for i, location in enumerate(points):
                        poes = hc_progress.result_matrix[i]
                        hc_data_inserter.add_entry(
                            hazard_curve_id=haz_curve.id,
                            poes=poes.tolist(),
                            location=location.wkt2d)

                    hc_data_inserter.flush()

    def post_process(self):
        """
        If requested by the user, compute mean hazard curves and save them to
        the database.
        """
        hc = self.job.hazard_calculation

        if hc.mean_hazard_curves:
            self._calculate_mean_curves()

        if len(hc.quantile_hazard_curves) > 0:
            self._calculate_quantile_curves()

    def _calculate_mean_curves(self):
        """
        For each intensity measure type, collect all of the curves for all
        logic tree realizations and calculate the mean/weighted average for
        each point of interest in the calculation.

        Resulting mean curve sets will be saved to the database, ready for
        subsequent export.
        """
        hc = self.job.hazard_calculation

        all_points = hc.points_to_compute()

        for imt, imls in hc.intensity_measure_types_and_levels.iteritems():
            im_type, sa_period, sa_damping = haz_general.split_imt_str(imt)

            hco = models.Output.objects.create(
                owner=hc.owner,
                oq_job=self.job,
                display_name='mean-curves-%s' % imt,
                output_type='hazard_curve')

            haz_curve = models.HazardCurve.objects.create(
                output=hco,
                investigation_time=hc.investigation_time,
                imt=im_type,
                imls=imls,
                sa_period=sa_period,
                sa_damping=sa_damping,
                statistics='mean')

            for point in all_points:
                # all hazard curves for this point and imt
                mean_curve = compute_mean_curve_for_point(
                    point, im_type, sa_period, sa_damping, self.job)

                models.HazardCurveData.objects.create(
                    hazard_curve=haz_curve,
                    poes=mean_curve,
                    location=point.wkt2d)

    def _calculate_quantile_curves(self):
        """
        For each intensity measure type, collect all of the curves for all
        logic tree realizations and calculate quantile or weighted quantile
        curves for each point of interest in the calculation. Many quantile
        levels can be specified.

        Resulting quantile curve sets will be saved to the database, ready for
        subsequent export.
        """
        hc = self.job.hazard_calculation

        # Were the hazard curves computed using the Monte-Carlo approach?
        # Or end-branch enumeration?
        # This determines which quantile calculation method we apply.
        enumerated = hc.number_of_logic_tree_samples == 0

        all_points = hc.points_to_compute()

        for imt, imls in hc.intensity_measure_types_and_levels.iteritems():
            im_type, sa_period, sa_damping = haz_general.split_imt_str(imt)

            q_curve_sets = dict()
            for quantile in hc.quantile_hazard_curves:
                hco = models.Output.objects.create(
                    owner=hc.owner,
                    oq_job=self.job,
                    display_name=('quantile(%(quantile)s)-curves-%(imt)s'
                                  % dict(quantile=quantile, imt=imt)),
                    output_type='hazard_curve')

                haz_curve = models.HazardCurve.objects.create(
                    output=hco,
                    investigation_time=hc.investigation_time,
                    imt=im_type,
                    imls=imls,
                    sa_period=sa_period,
                    sa_damping=sa_damping,
                    statistics='quantile',
                    quantile=quantile)

                q_curve_sets[quantile] = haz_curve

            for point in all_points:
                curves, weights = curves_weights_for_point(
                    point, im_type, sa_period, sa_damping, self.job)

                for quantile in hc.quantile_hazard_curves:
                    # all hazard curves for this point, imt, and quantile
                    # TODO: compute quantile curve
                    # TODO: save quantile curve to DB
                    if enumerated:
                        q_curve = compute_weighted_quantile_curve(
                            curves, weights, quantile)
                        # TODO: save me
                    else:
                        q_curve = compute_quantile_curve(curves, quantile)
                        # TODO: save me

                    models.HazardCurveData.objects.create(
                        hazard_curve=q_curve_sets[quantile],
                        poes=q_curve,
                        location=point.wkt2d)

    def clean_up(self):
        """
        Delete temporary database records. These records represent intermediate
        copies of final calculation results and are no longer needed.

        In this case, this includes all of the data for this calculation in the
        tables found in the `htemp` schema space.
        """
        hc = self.job.hazard_calculation

        logs.LOG.debug('> cleaning up temporary DB data')
        models.HazardCurveProgress.objects.filter(
            lt_realization__hazard_calculation=hc.id).delete()
        models.SourceProgress.objects.filter(
            lt_realization__hazard_calculation=hc.id).delete()
        models.SiteData.objects.filter(hazard_calculation=hc.id).delete()
        logs.LOG.debug('< done cleaning up temporary DB data')


def curves_weights_for_point(point, imt, sa_period, sa_damping, job):
    """
    Fetch the sequence of hazard curve PoE values (as a list of lists of
    floats) for a given ``point``, ``imt``, and ``job``. Also return a list of
    `weight` values for each curve. (NOTE: For Monte-Carlo logic tree sampling,
    the weights are implicit and will all be `None`. For logic tree end branch
    enumeration, weights are explicit.)

    :param point:
        :class:`nhlib.geo.point.Point` instance indicating the point of
        interest for mean curve computation. This location will be used
        in spatial query to fetch input data for the mean curve computation.
    :param str imt:
        Intensity measure type (PGA, SA, PGV, etc.).
    :param float sa_period:
        Only relevant if ``imt`` is "SA".
    :param float sa_damping:
        Only relevant if ``imt`` is "SA".
    :param job:
        :class:`openquake.db.models.OqJob` instance.

    :returns:
        A double of (curves, weights). Both should be of equal length.
    """

    curves_for_point = models.HazardCurveData.objects.filter(
        hazard_curve__output__oq_job=job,
        hazard_curve__imt=imt,
        hazard_curve__sa_period=sa_period,
        hazard_curve__sa_damping=sa_damping,
        # We only want curves associated with a logic tree
        # realization
        hazard_curve__lt_realization__isnull=False,
        location__equals=point.wkt2d).select_related(
            # fetch the lt_realization info as well, to get the
            # weights as well in just a single query
            'hazard_curve__lt_realization')

    curves_poes = [crv.poes for crv in curves_for_point]
    curves_weights = [crv.hazard_curve.lt_realization.weight
                      for crv in curves_for_point]

    return curves_poes, curves_weights


def compute_mean_curve_for_point(point, imt, sa_period, sa_damping, job):
    """
    Compute and return the mean or weighted average curve for a given
    ``point``, ``imt``, and ``job``.

    :param point:
        :class:`nhlib.geo.point.Point` instance indicating the point of
        interest for mean curve computation. This location will be used
        in spatial query to fetch input data for the mean curve computation.
    :param str imt:
        Intensity measure type (PGA, SA, PGV, etc.).
    :param float sa_period:
        Only relevant if ``imt`` is "SA".
    :param float sa_damping:
        Only relevant if ``imt`` is "SA".
    :param job:
        :class:`openquake.db.models.OqJob` instance.

    :returns:
        1D numpy array representing the mean curve.
    """
    curves, weights = curves_weights_for_point(
        point, imt, sa_period, sa_damping, job)

    return compute_mean_curve(curves, weights)


def compute_mean_curve(curves, weights=None):
    """
    Compute the mean or weighted average of a set of curves.

    :param curves:
        2D array-like collection of hazard curve PoE values. Each element
        should be a sequence of PoE `float` values. Example::

            [[0.5, 0.4, 0.3], [0.6, 0.59, 0.1]]
    :param weights:
        List or numpy array of weights, 1 weight value for each of the input
        ``curves``. This is only used for weighted averages.

    :returns:
        A curve representing the mean/average (or weighted average, in case
        ``weights`` are specified) of all the input ``curves``.
    """
    # Weights
    if weights is not None:
        # If all of the weights are None, don't compute a weighted average
        if set(weights) == set([None]):
            weights = None
        elif any([x is None for x in weights]):
            # a subset of the weights are None
            # this is invalid
            raise ValueError('`None` value found in weights: %s' % weights)

    return numpy.average(curves, weights=weights, axis=0)


# TODO: possible future optimization: compute many quantiles at once
def compute_quantile_curve(curves, quantile):
    """
    Compute the quantile aggregate of a set of curves. This method is used in
    the case where hazard curves are computed using the Monte-Carlo logic tree
    sampling approach. In this case, the weights are implicit.

    :param curves:
        2D array-like collection of hazard curve PoE values. Each element
        should be a sequence of PoE `float` values. Example::

            [[0.5, 0.4, 0.3], [0.6, 0.59, 0.1]]
    :param float quantile:
        The quantile value. We expected a value in the range [0.0, 1.0].

    :returns:
        A numpy array representing the quantile aggregate of the input
        ``curves`` and ``quantile``.
    """
    return numpy.array(mstats.mquantiles(curves, prob=quantile, axis=0))[0]


# TODO: possible future optimization: compute many quantiles as once
def compute_weighted_quantile_curve(curves, weights, quantile):
    """
    Compute the weighted quantile aggregate of a set of curves. This method is
    used in the case where hazard curves are computed using the logic tree
    end-branch enumeration approach. In this case, the weights are explicit.

    TODO(LB): Link documentation about weighted quantile calculation. From what
    I understand, this method is not very obvious. A quick websearch came up
    with no documentation which was immediately useful.

    :param curves:
        2D array-like of curve PoEs. Each row represents the PoEs for a single
        curve
    :param weights:
        Array-like of weights, 1 for each input curve.
    :param quantile:
        Quantile value to calculate. Should in the range [0.0, 1.0].

    :returns:
        A numpy array representing the quantile aggregate of the input
        ``curves`` and ``quantile``, weighting each curve with the specified
        ``weights``.
    """
    # Each curve needs to be associated with a weight:
    assert len(weights) == len(curves)

    result_curve = []

    np_curves = numpy.array(curves)
    np_weights = numpy.array(weights)

    for poes in np_curves.transpose():
        sorted_poe_idxs = numpy.argsort(poes)
        sorted_weights = np_weights[sorted_poe_idxs]
        sorted_poes = poes[sorted_poe_idxs]

        # cumulatative sum of weights:
        cum_weights = numpy.cumsum(sorted_weights)

        result_curve.append(numpy.interp(quantile, cum_weights, sorted_poes))

    return numpy.array(result_curve)


def update_result_matrix(current, new):
    """
    Use the following formula to combine multiple iterations of results:

    `result = 1 - (1 - current) * (1 - new)`

    This is used to incrementally update hazard curve results by combining an
    initial value with some new results. (Each set of new results is computed
    over only a subset of seismic sources defined in the calculation model.)

    Parameters are expected to be multi-dimensional numpy arrays, but the
    formula will also work with scalars.

    :param current:
        Numpy array representing the current result matrix value.
    :param new:
        Numpy array representing the new results which need to be combined with
        the current value. This should be the same shape as `current`.
    """
    return 1 - (1 - current) * (1 - new)
