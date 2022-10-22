# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing

from .util import estimation

logger = logging.getLogger(__name__)

@inject.step()
def plays_pokemon(persons_merged, persons, chunk_size):
    """ """

    trace_label = "plays_pokemon"
    model_settings_file_name = "plays_pokemon.yaml"

    choosers = persons_merged.to_frame()
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    model_settings = config.read_model_settings(model_settings_file_name)
    constants = config.get_model_constants(model_settings)

    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, False)

    nest_spec = config.get_logit_model_settings(model_settings)

    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name="person_plays_pokemon",
        estimator=False,
    )

    plays_pokemon_alt = model_settings["PLAYS_POKEMON_ALT"]
    choices = choices == plays_pokemon_alt

    persons = persons.to_frame()
    persons["plays_pokemon"] = (
        choices.reindex(persons.index).fillna(0).astype(bool)
    )

    pipeline.replace_table("persons", persons)

    tracing.print_summary(
        "plays_pokemon", persons.free_parking_at_work, value_counts=True
    )