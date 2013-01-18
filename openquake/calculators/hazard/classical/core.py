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

import nhlib
import nhlib.calc
import nhlib.imt

import numpy

import openquake

from django.db import transaction

from openquake import logs
from openquake.calculators import base
from openquake.calculators.hazard import general as haz_general
from openquake.db import models
from openquake.input import logictree
from openquake.utils import stats
from openquake.utils import tasks as utils_tasks
from openquake.utils.general import block_splitter as bs

from openquake.db.aggregate_result_writer import (MeanCurveWriter,
                                                  QuantileCurveWriter)
from openquake.calculators.hazard.classical import post_processing


@utils_tasks.oqtask
@stats.count_progress('h')
def hazard_curves(job_id, src_ids, lt_rlz_id):
    """
    A celery task wrapper function around :func:`compute_hazard_curves`.
    See :func:`compute_hazard_curves` for parameter definitions.
    """
    logs.LOG.debug('> starting task: job_id=%s, lt_realization_id=%s'
                   % (job_id, lt_rlz_id))

    compute_hazard_curves(job_id, src_ids, lt_rlz_id)
    # Last thing, signal back the control node to indicate the completion of
    # task. The control node needs this to manage the task distribution and
    # keep track of progress.
    logs.LOG.debug('< task complete, signalling completion')
    base.signal_task_complete(job_id=job_id, num_items=len(src_ids))


# Silencing 'Too many local variables'
# pylint: disable=R0914
def compute_hazard_curves(job_id, src_ids, lt_rlz_id):
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
        haz_general.update_realization(lt_rlz.id, len(src_ids))

    logs.LOG.debug('< transaction complete')


class ClassicalHazardCalculator(haz_general.BaseHazardCalculatorNext):
    """
    Classical PSHA hazard calculator. Computes hazard curves for a given set of
    points.

    For each realization of the calculation, we randomly sample source models
    and GMPEs (Ground Motion Prediction Equations) from logic trees.
    """

    core_calc_task = hazard_curves

    def task_arg_gen(self, block_size):
        """
        Loop through realizations and sources to generate a sequence of
        task arg tuples. Each tuple of args applies to a single task.

        Yielded results are triples of (job_id, realization_id,
        source_id_list).

        :param int block_size:
            The (max) number of work items for each each task. In this case,
            sources.
        """
        realizations = models.LtRealization.objects.filter(
                hazard_calculation=self.hc, is_complete=False)

        for lt_rlz in realizations:
            source_progress = models.SourceProgress.objects.filter(
                    is_complete=False, lt_realization=lt_rlz).order_by('id')
            source_ids = source_progress.values_list('parsed_source_id',
                                                     flat=True)

            for offset in xrange(0, len(source_ids), block_size):
                task_args = (
                    self.job.id,
                    source_ids[offset:offset + block_size],
                    lt_rlz.id
                )
                yield task_args

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

        self.record_init_stats()

        # Set the progress counters:
        num_sources = models.SourceProgress.objects.filter(
            is_complete=False,
            lt_realization__hazard_calculation=self.hc).count()
        self.progress['total'] = num_sources

        self.initialize_pr_data()

    def post_execute(self):
        """
        Post-execution actions. At the moment, all we do is finalize the hazard
        curve results. See
        :meth:`openquake.calculators.hazard.general.BaseHazardCalculatorNext.finalize_hazard_curves`
        for more info.
        """
        self.finalize_hazard_curves()

    def clean_up(self):
        """
        Delete temporary database records. These records represent intermediate
        copies of final calculation results and are no longer needed.

        In this case, this includes all of the data for this calculation in the
        tables found in the `htemp` schema space.
        """
        logs.LOG.debug('> cleaning up temporary DB data')
        models.HazardCurveProgress.objects.filter(
            lt_realization__hazard_calculation=self.hc.id).delete()
        models.SourceProgress.objects.filter(
            lt_realization__hazard_calculation=self.hc.id).delete()
        models.SiteData.objects.filter(hazard_calculation=self.hc.id).delete()
        logs.LOG.debug('< done cleaning up temporary DB data')

    def post_process(self):
        logs.LOG.debug('> starting post processing')

        if self.hc.mean_hazard_curves:
            self._mean_post_proc()
        return


        # If `mean_hazard_curves` is True and/or `quantile_hazard_curves`
        # has some value (not an empty list), do post processing.
        # Otherwise, just skip it altogether.
        if self.hc.mean_hazard_curves or self.hc.quantile_hazard_curves:
            tasks = post_processing.setup_tasks(
                self.job, self.job.hazard_calculation,
                curve_finder=models.HazardCurveData.objects,
                writers=dict(mean_curves=MeanCurveWriter,
                             quantile_curves=QuantileCurveWriter))

            if openquake.no_distribute():
                # Don't use celery for the post-processing computation:
                for each_task in tasks:
                    post_processing.do_post_process(self.job.id, each_task)
            else:
                utils_tasks.distribute(
                    post_processing.do_post_process,
                    ("post_processing_task", tasks),
                    tf_args=dict(job_id=self.job.id)
                )

        if len(self.hc.poes_hazard_maps) > 0:
            post_processing.do_hazard_map_post_process(self.job)

        logs.LOG.debug('< done with post processing')

    def _mean_post_proc(self):
        max_curves_per_task = 100000
        num_rlzs = models.LtRealization.objects.filter(
            hazard_calculation=self.hc).count()

        num_site_blocks_per_incr = int(max_curves_per_task) / int(num_rlzs)
        # NOTE(larsbutler): `slice_incr` must be a multiple of `num_rlzs`
        slice_incr = num_site_blocks_per_incr * num_rlzs  # unit: num records

        if not (slice_incr / float(num_rlzs)) % 1 == 0:
            raise ValueError(
                "`slice_incr` is not evenly divisible by the number of "
                "realizations; something is very wrong"
            )

        for imt, imls in self.hc.intensity_measure_types_and_levels.items():
            im_type, sa_period, sa_damping = models.parse_imt(imt)

            hco = models.Output.objects.create(
                owner=self.hc.owner,
                oq_job=self.job,
                display_name='mean-curves-%s' % imt,
                output_type='hazard_curve'
            )
            haz_curve = models.HazardCurve.objects.create(
                output=hco,
                investigation_time=self.hc.investigation_time,
                imt=im_type,
                imls=imls,
                sa_period=sa_period,
                sa_damping=sa_damping,
                statistics='mean'
            )

            all_curves_for_imt = models.HazardCurveData.objects\
                    .filter(hazard_curve__output__oq_job=self.job,
                            hazard_curve__imt=im_type,
                            hazard_curve__sa_period=sa_period,
                            hazard_curve__sa_damping=sa_damping,
                            # We only want curves associated with a logic tree
                            # realization (and not statistical aggregates):
                            hazard_curve__lt_realization__isnull=False)\
                    .select_related('hazard_curve__lt_realization')\
                    .order_by('location')

            for chunk in queryset_iter(all_curves_for_imt, slice_incr):
                # slice each chunk by `num_rlzs` into `site_chunk`
                # and compute the aggregate
                for site_chunk in bs(chunk, num_rlzs):
                    site = site_chunk[0].location
                    curves_poes = [x.poes for x in site_chunk]
                    curves_weights = [x.hazard_curve.lt_realization.weight
                                      for x in site_chunk]
                    mean_curve = compute_mean_curve(
                        curves_poes, weights=curves_weights
                    )
                    models.HazardCurveData.objects.create(
                        hazard_curve=haz_curve,
                        poes=mean_curve,
                        location=site
                    )
            #chunks = list(queryset_iter(all_curves_for_imt, slice_incr))


def queryset_iter(queryset, chunk_size):
    offset = 0
    while True:
        chunk = queryset[offset:offset + chunk_size]
        if len(chunk) == 0:
            raise StopIteration
        else:
            yield chunk
            offset += chunk_size


def compute_mean_curve(curves, weights=None):
    """
    Compute the mean or weighted average of a set of curves.

    :param curves:
        2D array-like collection of hazard curve PoE values. Each element
        should be a sequence of PoE `float` values. Example::

            [[0.5, 0.4, 0.3], [0.6, 0.59, 0.1]]

        .. note::
            This data represents the curves for all realizations for a given
            site and IMT.

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
