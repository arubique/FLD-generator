#!/usr/bin/env python
import math
import random
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from pprint import pformat
import logging
from collections import defaultdict
import os
import sys

import click
from tqdm import tqdm
import dill


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

from FLD_generator.translators import build as build_translator, TemplatedTranslator
from FLD_generator.word_banks import build_wordbank
from FLD_generator.proof_tree_generation_pipeline import ProofTreeGenerationPipeline
from FLD_generator.proof_tree_generators import build as build_generator
from FLD_generator.datasets import NLProofSDataset
from FLD_generator.formula_distractors import build as build_distractor
from FLD_generator.translation_distractors import build as build_translation_distractor
from FLD_generator.utils import _build_bounded_msg, log_results, fix_seed
from FLD_generator.knowledge_banks import build_knowledge_bank
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor

from logger_setup import setup as setup_logger
sys.path.pop(0)


_USE_JOBLIB_FOR_PARALLEL = False


logger = logging.getLogger(__name__)


def load_dataset(
    argument_config: List[str],
    translation_lang: str,
    translation_config: List[str],
    use_fixed_translation: bool,
    reused_object_nouns_max_factor: float,
    limit_vocab_size_per_type: Optional[int],
    translation_volume_to_weight: str,
    translation_default_weight_factor_type: str,
    translation_adj_verb_noun_ratio: str,
    translation_no_transitive_object: bool,
    translation_vocab: str,
    complex_formula_arguments_weight: float,
    quantifier_axiom_arguments_weight: float,
    quantifier_axioms: str,
    quantification_degree: str,
    propositional_arguments_factor: float,
    negation_arguments_weight: float,
    theorem_tree_prob: float,
    theorem_arguments_factor: float,
    # adjust_theorem_argument_weight: bool,
    theorem_arguments_weight_adjustment_subset: str,
    knowledge_argument_factor: float,
    keep_dneg: bool,
    distractor: str,
    distractors_range: Tuple[int, int],
    sample_distractor_prototype_formulas_from_all_possible_formulas: bool,
    disallow_simplified_tree_formulas_as_distractor_prototype: bool,
    disallow_hard_negative_distractors: bool,
    # negative_tree_negated_hypothesis_ratio: float,
    disallow_subj_obj_swapped_distractor: bool,
    translation_distractor: str,
    fallback_from_formula_to_translation_distractor: bool,
    translation_distractors_range: Tuple[int, int],
    proof_stances: List[str],
    world_assump: str,
    unknown_ratio: float,
    reference_tree_prob: Optional[float],
    reference_argument_prob_in_depth_1: Optional[float],
    sample_all_stances_per_logic: bool,
    context_shuffles_per_instance: int,
    use_collapsed_translation_nodes_for_unknown_tree: bool,
    swap_ng_words: Optional[List[str]],

    generate_stem_steps_range: Tuple[int, int],
    generate_stem_steps_distrib: str,

    extend_branches_steps_range: Tuple[int, int],

    steps_limit: Optional[int],
    depth_limit: Optional[int],
    increase_depth_by_extend_branches: bool,

    force_fix_illegal_intermediate_constants: bool,
    distractor_variants_per_tree: int,
    translation_variants_per_logic: int,
    allow_smaller_proofs: bool,
    knowledge_range: float,
    collapsed_knowledge_range: float,
    knowledge_no_shuffle: bool,
    atomic_filepath: str,
    concept_net_100k_filepath: str,
    dbpedia_filepath: str,
):

    knowledge_banks = []
    if atomic_filepath is not None:
        knowledge_banks.append(
            build_knowledge_bank(
                'atomic',
                atomic_filepath,
                no_shuffle=knowledge_no_shuffle,
            )
        )
    if concept_net_100k_filepath is not None:
        knowledge_banks.append(
            build_knowledge_bank(
                'concept_net_100k',
                concept_net_100k_filepath,
                no_shuffle=knowledge_no_shuffle,
            )
        )
    if dbpedia_filepath is not None:
        knowledge_banks.append(
            build_knowledge_bank(
                'dbpedia',
                dbpedia_filepath,
                no_shuffle=knowledge_no_shuffle,
            )
        )

    generator = build_generator(
        argument_config,
        elim_dneg=not keep_dneg,
        complex_formula_arguments_weight=complex_formula_arguments_weight,
        quantifier_axiom_arguments_weight=quantifier_axiom_arguments_weight,
        quantifier_axioms=quantifier_axioms,
        quantification_degree=quantification_degree,
        propositional_arguments_factor=propositional_arguments_factor,
        negation_arguments_weight=negation_arguments_weight,
        theorem_tree_prob=theorem_tree_prob,
        theorem_arguments_factor=theorem_arguments_factor,
        # adjust_theorem_argument_weight=# adjust_theorem_argument_weight,
        theorem_arguments_weight_adjustment_subset=theorem_arguments_weight_adjustment_subset,
        knowledge_argument_factor=knowledge_argument_factor,
        knowledge_banks=knowledge_banks,
    )

    logger.info(_build_bounded_msg(f'{"[start] building wordnet":<30}', 3))
    word_bank = build_wordbank(
        translation_lang,
        extra_vocab = translation_vocab if translation_vocab not in [None, 'wordnet'] else None
    )
    logger.info(_build_bounded_msg(f'{"[finish] building wordnet":<30}', 3))

    if distractors_range[1] > 0:
        logger.info(_build_bounded_msg(f'{"[start] building distractor":<30}', 3))
        _distractor = build_distractor(
            distractor,
            generator=generator,
            sample_prototype_formulas_from_all_possible_formulas=sample_distractor_prototype_formulas_from_all_possible_formulas,
            disallow_simplified_formulas_as_prototype=disallow_simplified_tree_formulas_as_distractor_prototype,
            sample_hard_negatives=not disallow_hard_negative_distractors,
            # negative_tree_negated_hypothesis_ratio=negative_tree_negated_hypothesis_ratio,
        )
        logger.info(_build_bounded_msg(f'{"[finish] building distractor":<30}', 3))
    else:
        _distractor = None

    if translation_distractors_range[1] > 0:
        logger.info(_build_bounded_msg(f'{"[start] building translation distractor":<30}', 3))
        _translation_distractor = build_translation_distractor(
            translation_distractor,
            word_bank=word_bank,
            swap_ng_words=swap_ng_words,
        )
        logger.info(_build_bounded_msg(f'{"[finish] building translation distractor":<30}', 3))
    else:
        _translation_distractor = None

    logger.info(_build_bounded_msg(f'{"[start] building translator":<30}', 3))
    translator = build_translator(translation_lang,
                                  translation_config,
                                  word_bank,
                                  adj_verb_noun_ratio=translation_adj_verb_noun_ratio,
                                  use_fixed_translation=use_fixed_translation,
                                  reused_object_nouns_max_factor=reused_object_nouns_max_factor,
                                  limit_vocab_size_per_type=limit_vocab_size_per_type,
                                  volume_to_weight=translation_volume_to_weight,
                                  default_weight_factor_type=translation_default_weight_factor_type,
                                  knowledge_banks=knowledge_banks,
                                  no_transitive_object=translation_no_transitive_object)
    logger.info(_build_bounded_msg(f'{"[finish] building translator":<30}', 3))

    if translation_lang == 'eng':
        assumption_prefix = 'Let\'s assume that '
    elif translation_lang == 'jpn':
        assumption_prefix = '以下のように仮定する。'
    else:
        raise NotImplementedError()
    pipeline = ProofTreeGenerationPipeline(
        generator,
        distractor=_distractor,
        translation_distractor=_translation_distractor,
        fallback_from_formula_to_translation_distractor=fallback_from_formula_to_translation_distractor,
        translator=translator,
        assumption_prefix=assumption_prefix,
        add_subj_obj_swapped_distractor=not disallow_subj_obj_swapped_distractor,
        knowledge_range=knowledge_range,
        collapsed_knowledge_range=collapsed_knowledge_range,
    )

    if generate_stem_steps_distrib == 'flat':
        generate_stem_steps_weights = None
        _reference_argument_prob_in_depth_1 = reference_argument_prob_in_depth_1 or None
    elif generate_stem_steps_distrib == 'flat.no_reference':
        generate_stem_steps_weights = None
        _reference_argument_prob_in_depth_1 = reference_argument_prob_in_depth_1 or 0.0
    elif generate_stem_steps_distrib == 'ruletaker.ours.20221202':
        if set(generate_stem_steps_range) != (1, 3):
            raise ValueError(f'depths {generate_stem_steps_range} is not consistent with ruletaker.ours.20221202.')
        # see "depth distribution" of experiments.md
        generate_stem_steps_weights = [0.40, 0.15, 0.12]
        _reference_argument_prob_in_depth_1 = reference_argument_prob_in_depth_1 or 0.23 / (0.23 + 0.17)
    else:
        raise ValueError(f'Unknown depth distrib {generate_stem_steps_distrib}')

    return NLProofSDataset(
        pipeline,

        generate_stem_steps_range=generate_stem_steps_range,
        generate_stem_steps_weights=generate_stem_steps_weights,

        extend_branches_steps_range=extend_branches_steps_range,
        extend_branches_steps_weights=None,

        steps_limit=steps_limit,
        depth_limit=depth_limit,
        increase_depth_by_extend_branches=increase_depth_by_extend_branches,

        proof_stances=proof_stances,
        world_assump=world_assump,
        reference_tree_prob=reference_tree_prob,
        reference_argument_prob_in_depth_1=_reference_argument_prob_in_depth_1,
        force_fix_illegal_intermediate_constants=force_fix_illegal_intermediate_constants,
        distractors_range=distractors_range,
        translation_distractors_range=translation_distractors_range,
        unknown_ratio=unknown_ratio,
        sample_all_stances_per_logic=sample_all_stances_per_logic,
        context_shuffles_per_instance=context_shuffles_per_instance,
        use_collapsed_translation_nodes_for_unknown_tree=use_collapsed_translation_nodes_for_unknown_tree,
        swap_ng_words=swap_ng_words,
        word_bank = word_bank if use_collapsed_translation_nodes_for_unknown_tree else None,
        distractor_variants_per_tree=distractor_variants_per_tree,
        translation_variants_per_logic=translation_variants_per_logic,
        allow_smaller_proofs=allow_smaller_proofs,
    )


def generate_instances(size: int, *args):
    dataset = load_dataset(*args)
    data = []
    agg_stats = defaultdict(int)
    for i_sample, (nlproof_json, proof_tree, distractors, translation_distractors, stats) in tqdm(enumerate(dataset.generate(size))):
        data.append((nlproof_json, proof_tree, distractors, translation_distractors))

        log_results(logger, i_sample=i_sample, nlproof_json=nlproof_json, proof_tree=proof_tree,
                    distractors=distractors, translation_distractors=translation_distractors,
                    stats=None)

        if stats is not None:
            for name, count in stats.items():
                if count is not None:
                    # from pprint import pformat
                    # logger.critical('------------------------------------ agg_stats ------------------------------------')
                    # logger.critical(pformat(agg_stats))
                    agg_stats[name] = count   # !! should be equal, as stats already aggregated

    return data, agg_stats


@click.command()
@click.argument('output-path')
@click.argument('size', type=int)
@click.option('--argument-config', '--ac',
              multiple=True,
              default=[],
              help='argument (deduction rule) configuration files')
@click.option('--complex-formula-arguments-weight', type=float, default=0.5)
@click.option('--quantifier-axiom-arguments-weight', type=float, default=0.2)
# @click.option('--quantifier-axiom', multiple=True, default=None)
@click.option('--quantifier-axiom', type=str, default='all')
@click.option('--quantification-degree', type=str, default='all_constants')
@click.option('--propositional-arguments-factor', type=float, default=1.0)
@click.option('--negation-arguments-weight', type=float, default=None,
              help="""Negation arguments can affect the benchmark performance, especially benchmarks such as NLI,
                      as they require distinguishing between 'entailment' and 'contradiction'.
                      So we should fix their ratio for fair comparison among different corpora.""")
@click.option('--theorem-tree-prob', type=float, default=1.0,
              help="""The probability of trees that 'allow' theorem arguments to be included.
                      Note that it does not mean that these trees always include theorem arguments.
                      Whether to include theorem arguments or not is determined by random sampling,
                      which mainly affected by 'theorem-arguments-factor""")
@click.option('--theorem-arguments-factor', type=float, default=0.3)
# @click.option('--adjust-theorem-argument-weight', type=bool, is_flag=True, default=False,
#               help='If True, the weights for more important theorem arguments are increased.')
@click.option('--theorem-arguments-weight-adjustment-subset', type=str, default=None)
#
@click.option('--knowledge-argument-factor', type=float, default=1.0)
#
@click.option('--generate-stem-steps-range', type=str, default=json.dumps([1, 5]))
@click.option('--generate-stem-steps-distrib', default='flat', type=click.Choice(['flat', 'flat.no_reference', 'ruletaker.ours.20221202']))
@click.option('--extend-branches-steps-range', type=str, default=json.dumps([5, 5]))
@click.option('--steps-limit', type=int, default=None)
@click.option('--depth-limit', type=int, default=None)
@click.option('--increase-depth-by-extend-branches', is_flag=True)
#
@click.option('--force-fix-illegal-intermediate-constants', is_flag=True)
@click.option('--keep-dneg', is_flag=True, default=False)
#
@click.option('--translation-lang', type=str, default='eng')
@click.option('--translation-config', '--tc',
              multiple=True,
              default=['./configs/translations/thing.v1'],
              help='natural language translation config files')
@click.option('--use-fixed-translation', type=bool, is_flag=True)
@click.option('--reused-object-nouns-max-factor', type=float, default=1.0)
@click.option('--limit-vocab-size-per-type', type=int, default=None)
@click.option('--translation-volume-to-weight', type=str, default='log10')
@click.option('--translation-default-weight-factor-type', type=str, default='W_VOL__1.0')
@click.option('--translation-adj-verb-noun-ratio', type=str, default='1-1-1')
@click.option('--translation-no-transitive-object', type=bool, is_flag=True)
@click.option('--translation-vocab', type=str, default=None)
#
# @click.option('--distractor', default='mixture.negative_tree.negative_tree')
@click.option('--distractor', default='mixture(negative_tree_double.simplified_formula.various_form)')
@click.option('--distractors-range', type=str, default=json.dumps([0, 20]))
@click.option('--disallow-hard-negative-distractors', type=bool, is_flag=True)
# @click.option('--negative-tree-negated-hypothesis-ratio', type=float, default=0.5)
@click.option('--sample-distractor-prototype-formulas-from-all-possible-formulas', type=bool, is_flag=True)
@click.option('--disallow-simplified-tree-formulas-as-distractor-prototype', type=bool, is_flag=True)
@click.option('--disallow-subj-obj-swapped-distractor', type=bool, is_flag=True)
@click.option('--translation-distractor', default='word_swap')
@click.option('--translation-distractors-range', type=str, default=json.dumps([0, 0]),
              help='SHOULD NOT USE, as it can lead to logically inconsistent facts.')
@click.option('--fallback-from-formula-to-translation-distractor', is_flag=True, default=False)
#
@click.option('--knowledge-range', type=str, default=json.dumps([0.0, 0.0]))
@click.option('--collapsed-knowledge-range', type=str, default=json.dumps([0.0, 0.0]))
@click.option('--knowledge-no-shuffle', is_flag=True, type=bool, default=False)
@click.option('--atomic-filepath', type=str, default=None)
@click.option('--concept-net-100k-filepath', type=str, default=None)
@click.option('--dbpedia-filepath', type=str, default=None)
#
@click.option('--proof-stances', type=str, default=json.dumps(['PROVED', 'DISPROVED', 'UNKNOWN']))
@click.option('--world-assump', default='OWA')
@click.option('--unknown-ratio', type=float, default = 1 / 3.)
@click.option('--reference-tree-prob', type=float, default = None,
              help='''Reference arguments affect the benchmark performance signififancly,
                      so we should fix their ratio for fair comparison among different corpora.''')
@click.option('--reference-argument-prob-in-depth-1', type=float, default = None)
@click.option('--sample-all-stances-per-logic', is_flag=True, default=False,
              help='Augmentation. But, it seems to be better to just increase the size of the dataset without augmentation, if computationally feasible.')
@click.option('--context-shuffles-per-instance', type=int, default=1,
              help='Augmentation. Same comment as "sample-all-stances-per-logic."')
@click.option('--use-collapsed-translation-nodes-for-unknown-tree', is_flag=True, default=False)
@click.option('--swap-ng-words-config', default=None)
#
@click.option('--distractor-variants-per-tree', type=int, default=1,
              help='Augmentation. Same comment as "sample-all-stances-per-logic."')
@click.option('--translation-variants-per-logic', type=int, default=1,
              help='Augmentation. Same comment as "sample-all-stances-per-logic."')
#
@click.option('--allow-smaller-proofs', is_flag=True, default=False)
#
@click.option('--num-workers', type=int, default=1)
@click.option('--min-size-per-worker', type=int,
              default=10,
              # multithread  : data load = 4min, generation = 140 instances / 14min = 10 instances / min
              )
@click.option('--batch-size-per-worker', type=int, default=10000)
@click.option('--seed', type=int, default=0)
def main(output_path,
         argument_config,
         translation_lang,
         translation_config,
         use_fixed_translation,
         reused_object_nouns_max_factor,
         limit_vocab_size_per_type,
         translation_volume_to_weight,
         translation_default_weight_factor_type,
         translation_adj_verb_noun_ratio,
         translation_no_transitive_object,
         translation_vocab,
         size,

         generate_stem_steps_range,
         generate_stem_steps_distrib,
         extend_branches_steps_range,
         steps_limit,
         depth_limit,
         increase_depth_by_extend_branches,

         force_fix_illegal_intermediate_constants,
         complex_formula_arguments_weight,
         quantifier_axiom_arguments_weight,
         quantifier_axiom,
         quantification_degree,
         propositional_arguments_factor,
         negation_arguments_weight,
         theorem_tree_prob,
         theorem_arguments_factor,
         # adjust_theorem_argument_weight,
         theorem_arguments_weight_adjustment_subset,
         knowledge_argument_factor,
         keep_dneg,
         distractor,
         distractors_range,
         sample_distractor_prototype_formulas_from_all_possible_formulas,
         disallow_simplified_tree_formulas_as_distractor_prototype,
         disallow_hard_negative_distractors,
         # negative_tree_negated_hypothesis_ratio,
         disallow_subj_obj_swapped_distractor,
         translation_distractor,
         fallback_from_formula_to_translation_distractor,
         translation_distractors_range,
         knowledge_range,
         collapsed_knowledge_range,
         knowledge_no_shuffle,
         atomic_filepath,
         concept_net_100k_filepath,
         dbpedia_filepath,
         proof_stances,
         world_assump,
         unknown_ratio,
         reference_tree_prob,
         reference_argument_prob_in_depth_1,
         sample_all_stances_per_logic,
         context_shuffles_per_instance,
         use_collapsed_translation_nodes_for_unknown_tree,
         swap_ng_words_config,
         distractor_variants_per_tree,
         translation_variants_per_logic,
         allow_smaller_proofs,
         num_workers,
         min_size_per_worker,
         batch_size_per_worker,
         seed):
    setup_logger(do_stderr=True, level=logging.INFO)
    fix_seed(seed)
    generate_stem_steps_range = tuple(json.loads(generate_stem_steps_range))
    extend_branches_steps_range = json.loads(extend_branches_steps_range)
    distractors_range = json.loads(distractors_range)
    translation_distractors_range = json.loads(translation_distractors_range)
    knowledge_range = json.loads(knowledge_range)
    collapsed_knowledge_range = json.loads(collapsed_knowledge_range)
    proof_stances = json.loads(proof_stances)
    swap_ng_words = json.load(open(swap_ng_words_config)) if swap_ng_words_config is not None else None

    if len(argument_config) == 0:
        raise ValueError()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    size_per_worker = math.ceil(size / num_workers)
    if size_per_worker < min_size_per_worker:
        num_workers = max(int(size / min_size_per_worker), 1)
    size_per_worker = math.ceil(size / num_workers)

    _batch_size_per_worker = min(batch_size_per_worker, size_per_worker)
    num_batches = math.ceil(size_per_worker / _batch_size_per_worker)

    logger.info('num_workers: %d', num_workers)
    logger.info('size_per_worker: %d', size_per_worker)
    logger.info('batch_size_per_worker: %d', _batch_size_per_worker)
    logger.info('num_batches: %d', num_batches)

    gathered_stats = defaultdict(int)
    with open(output_path, 'w') as f_out:

        for i_batch in range(num_batches):
            job_args = []
            for _ in range(num_workers):
                job_args.append([
                    _batch_size_per_worker,
                    argument_config,
                    translation_lang,
                    translation_config,
                    use_fixed_translation,
                    reused_object_nouns_max_factor,
                    limit_vocab_size_per_type,
                    translation_volume_to_weight,
                    translation_default_weight_factor_type,
                    translation_adj_verb_noun_ratio,
                    translation_no_transitive_object,
                    translation_vocab,
                    complex_formula_arguments_weight,
                    quantifier_axiom_arguments_weight,
                    quantifier_axiom,
                    quantification_degree,
                    propositional_arguments_factor,
                    negation_arguments_weight,
                    theorem_tree_prob,
                    theorem_arguments_factor,
                    # adjust_theorem_argument_weight,
                    theorem_arguments_weight_adjustment_subset,
                    knowledge_argument_factor,
                    keep_dneg,
                    distractor,
                    distractors_range,
                    sample_distractor_prototype_formulas_from_all_possible_formulas,
                    disallow_simplified_tree_formulas_as_distractor_prototype,
                    disallow_hard_negative_distractors,
                    disallow_subj_obj_swapped_distractor,
                    translation_distractor,
                    fallback_from_formula_to_translation_distractor,
                    translation_distractors_range,
                    proof_stances,
                    world_assump,
                    unknown_ratio,
                    reference_tree_prob,
                    reference_argument_prob_in_depth_1,
                    sample_all_stances_per_logic,
                    context_shuffles_per_instance,
                    use_collapsed_translation_nodes_for_unknown_tree,
                    swap_ng_words,

                    generate_stem_steps_range,
                    generate_stem_steps_distrib,

                    extend_branches_steps_range,

                    steps_limit,
                    depth_limit,
                    increase_depth_by_extend_branches,

                    force_fix_illegal_intermediate_constants,
                    distractor_variants_per_tree,
                    translation_variants_per_logic,
                    allow_smaller_proofs,
                    knowledge_range,
                    collapsed_knowledge_range,
                    knowledge_no_shuffle,
                    atomic_filepath,
                    concept_net_100k_filepath,
                    dbpedia_filepath,
                ])


            cnt = 0
            is_done = False
            num_jobs: Dict[str, int] = defaultdict(int)

            def update(instance, stats):
                nonlocal cnt
                nonlocal is_done
                nonlocal num_jobs

                if is_done:
                    return

                for nlproof_json, proof_tree, _, _ in instances:
                    if cnt >= size:
                        is_done = True
                        break
                    f_out.write(json.dumps(nlproof_json) + '\n')
                    cnt += 1

                for name, count in stats.items():
                    if count is not None:
                        # from pprint import pformat
                        # logger.critical('------------------------------------ gathered stats master ------------------------------------')
                        # logger.critical(pformat(gathered_stats))
                        gathered_stats[name] += count
                        num_jobs[name] += 1

            logger.info('creating corpus with %d jobs', num_workers)

            if _USE_JOBLIB_FOR_PARALLEL:
                jobs = [delayed(generate_instances)(*args) for job_args in job_args]
                job_results = Parallel(n_jobs=num_workers, backend='multiprocessing')(jobs)
                for instances, stats in job_results:
                    update(instances, stats)
            else:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(generate_instances, *args) for args in job_args]
                    for future in futures:
                        instances, stats = future.result()
                        update(instances, stats)

            for name, count in gathered_stats.items():
                if not name.startswith('cum.'):
                    gathered_stats[name] = gathered_stats[name] / num_jobs[name]

            logger.info('=========================== gathered stats (batch=%d) ============================',
                        i_batch)
            logger.info('\n' + pformat(gathered_stats))

    with open(str(output_path) + '.stats.json', 'w') as f_out:
        json.dump(dict(gathered_stats), f_out,
                  ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    logger.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! create_FLD_corpus.py DONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


if __name__ == '__main__':
    main()
