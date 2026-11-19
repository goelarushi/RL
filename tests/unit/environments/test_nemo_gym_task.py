# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy

import pytest

from nemo_rl.environments.nemo_gym_request import (
    _chat_template_kwargs_for_processor,
    _metadata_extra_body,
)
from nemo_rl.environments.nemo_gym_task import (
    get_nemo_gym_attribution_key,
    get_nemo_gym_responses_create_params,
    get_nemo_gym_task_input,
    is_nemo_gym_task,
)
from nemo_rl.experience.reward_penalties import is_low_effort
from nemo_rl.experience.rollouts import EffortLevelsConfig


def _harbor_row(*turns):
    return {
        "task_id": {"taskset": "agentic_vbench_train", "task_id": "t-1"},
        "task_input": {"task_data": {"family": "assembly"}, "turns": list(turns)},
    }


@pytest.mark.parametrize(
    "row,expected",
    [
        (
            {
                "task_id": {"taskset": "workplace:train"},
                "task_input": {},
                "agent_ref": {"name": "stale"},
                "attribution_key": "stale",
            },
            "workplace:train",
        ),
        (
            {"agent_ref": {"name": "actual_agent"}, "attribution_key": "stale"},
            "actual_agent",
        ),
        ({"attribution_key": "workplace:train"}, "workplace:train"),
        (
            {
                "task_id": {"taskset": ""},
                "task_input": {},
                "agent_ref": {"name": "stale"},
            },
            None,
        ),
        ({}, None),
    ],
)
def test_attribution_uses_taskset_or_actual_flat_agent(row, expected):
    assert get_nemo_gym_attribution_key(row) == expected


@pytest.mark.parametrize(
    "row",
    [
        {"responses_create_params": {}},
        {"task_id": "flat-id", "task_input": {"custom": True}},
        {"task_id": {"taskset": "workplace:train"}},
    ],
)
def test_nonmaterialized_task_fields_preserve_flat_row(row):
    assert not is_nemo_gym_task(row)
    assert get_nemo_gym_task_input(row) is row


@pytest.mark.parametrize("taskset", ["workplace:train", "", None])
def test_materialized_task_input_returns_nested_object(taskset):
    row = {
        "task_id": {"taskset": taskset, "task_id": "17"},
        "task_input": {
            "responses_create_params": {"input": "Find the next meeting."},
        },
    }

    assert is_nemo_gym_task(row)
    assert get_nemo_gym_task_input(row) is row["task_input"]


@pytest.mark.parametrize("task_input", [None, [], "serialized"])
def test_malformed_materialized_task_input_is_rejected(task_input):
    row = {
        "task_id": {"taskset": "workplace:train", "task_id": "17"},
        "task_input": task_input,
    }
    with pytest.raises(TypeError, match="materialized task_input must be a dict"):
        get_nemo_gym_task_input(row)


@pytest.mark.parametrize("shape", ["flat", "materialized", "harbor_turns"])
def test_request_metadata_is_read_without_rewriting_envelope(shape):
    task_input = {
        "responses_create_params": {
            "metadata": {
                "extra_body": '{"seed": 7, "chat_template_kwargs": {"enable_thinking": true}}',
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
        "task_data": {"expected_answer": "opaque"},
    }
    if shape == "flat":
        row = task_input
    elif shape == "materialized":
        row = {
            "task_id": {"taskset": "math:train", "task_id": "4"},
            "task_input": task_input,
        }
    else:
        row = _harbor_row(
            {
                "name": "solve",
                "responses_create_params": task_input["responses_create_params"],
            }
        )
    original = deepcopy(row)

    assert _metadata_extra_body(row) == {
        "seed": 7,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert _chat_template_kwargs_for_processor(row) == {
        "chat_template_kwargs": {"enable_thinking": False},
        "enable_thinking": False,
    }
    assert row == original


def test_responses_create_params_of_flat_and_materialized_rows():
    flat = {"responses_create_params": {"input": "hi"}}
    materialized = {"task_id": {"taskset": "s"}, "task_input": deepcopy(flat)}

    assert get_nemo_gym_responses_create_params(flat) is flat["responses_create_params"]
    assert (
        get_nemo_gym_responses_create_params(materialized)
        is materialized["task_input"]["responses_create_params"]
    )
    assert get_nemo_gym_responses_create_params({"agent_ref": {"name": "a"}}) is None


def test_responses_create_params_of_single_turn_harbor_row_is_the_turn_object():
    turn = {"name": "solve", "responses_create_params": {"input": "Assemble."}}
    row = _harbor_row(turn)

    assert get_nemo_gym_responses_create_params(row) is turn["responses_create_params"]
    assert (
        get_nemo_gym_responses_create_params(row, create=True)
        is turn["responses_create_params"]
    )


def test_create_inserts_parameters_where_they_belong():
    turn = {"name": "solve"}
    harbor = _harbor_row(turn)
    flat = {"agent_ref": {"name": "a"}}

    get_nemo_gym_responses_create_params(harbor, create=True)["metadata"] = {"k": 1}
    get_nemo_gym_responses_create_params(flat, create=True)["metadata"] = {"k": 2}

    assert turn["responses_create_params"] == {"metadata": {"k": 1}}
    assert "responses_create_params" not in harbor["task_input"]
    assert flat["responses_create_params"] == {"metadata": {"k": 2}}


def test_multi_turn_harbor_row_is_not_resolved():
    row = _harbor_row(
        {"name": "a", "responses_create_params": {"input": "1"}},
        {"name": "b", "responses_create_params": {"input": "2"}},
    )
    original = deepcopy(row)

    assert get_nemo_gym_responses_create_params(row) is None
    assert get_nemo_gym_responses_create_params(row, create=True) is None
    assert row == original


def test_low_effort_reads_the_single_turn_prompt():
    config = EffortLevelsConfig(low_weight=1.0, low_string="/low")
    row = _harbor_row(
        {
            "name": "solve",
            "responses_create_params": {
                "input": [{"role": "user", "content": "go /low"}]
            },
        }
    )

    assert is_low_effort(row, config)
    with pytest.raises(TypeError, match="responses_create_params dict"):
        is_low_effort(_harbor_row({"name": "a"}, {"name": "b"}), config)
