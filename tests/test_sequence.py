# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.sequence import IntermediateTensors


def test_sequence_intermediate_tensors_equal():
    class AnotherIntermediateTensors(IntermediateTensors):
        pass

    intermediate_tensors = IntermediateTensors({})
    another_intermediate_tensors = AnotherIntermediateTensors({})
    assert intermediate_tensors != another_intermediate_tensors

    empty_intermediate_tensors_1 = IntermediateTensors({})
    empty_intermediate_tensors_2 = IntermediateTensors({})
    assert empty_intermediate_tensors_1 == empty_intermediate_tensors_2

    different_key_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    difference_key_intermediate_tensors_2 = IntermediateTensors(
        {"2": torch.zeros([2, 4], dtype=torch.int32)}
    )
    assert different_key_intermediate_tensors_1 != difference_key_intermediate_tensors_2

    same_key_different_value_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    same_key_different_value_intermediate_tensors_2 = IntermediateTensors(
        {"1": torch.zeros([2, 5], dtype=torch.int32)}
    )
    assert (
        same_key_different_value_intermediate_tensors_1
        != same_key_different_value_intermediate_tensors_2
    )

    same_key_same_value_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    same_key_same_value_intermediate_tensors_2 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    assert (
        same_key_same_value_intermediate_tensors_1
        == same_key_same_value_intermediate_tensors_2
    )


def test_slice_for_pp_keeps_mixed_layouts_on_the_same_storage():
    hidden_states = torch.arange(48).reshape(16, 3)
    aux = torch.arange(32).reshape(16, 2)
    backing = IntermediateTensors({"hidden_states": hidden_states, "aux": aux})

    for num_tokens, shard_rows in ((16, 8), (9, 5), (1, 1), (16, 8)):
        result = backing.slice_for_pp(num_tokens, ("hidden_states",), 2)
        torch.testing.assert_close(result["hidden_states"], hidden_states[:shard_rows])
        torch.testing.assert_close(result["aux"], aux[:num_tokens])
        assert result["hidden_states"].data_ptr() == hidden_states.data_ptr()
        assert result["aux"].data_ptr() == aux.data_ptr()

    assert backing.slice_for_pp(9, (), 2) == backing[:9]
    torch.testing.assert_close(hidden_states, torch.arange(48).reshape(16, 3))
    torch.testing.assert_close(aux, torch.arange(32).reshape(16, 2))
