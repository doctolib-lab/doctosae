"""Parsing of Interpreter labels and Activation Prediction scores, as the paper's local backend did."""

from __future__ import annotations

import json
import math

import pytest

from doctosae.interpret.cards import PARSE_ERROR
from doctosae.interpret.labeling import Label, label_from_json, parse_label
from doctosae.interpret.scoring import parse_score


def test_label_after_reasoning_text_and_code_fence():
    payload = '{"topic": "bone", "description": "Bone grafts.", "keywords": ["bone"]}'
    reply = f"Shared theme: bone.\n```json\n{payload}\n```"
    assert parse_label(reply) == Label("bone", "Bone grafts.", ["bone"])


def test_last_valid_group_wins_and_earlier_groups_are_fallbacks():
    reply = '{"topic": "first"} then {"topic": "second", "keywords": []}'
    assert parse_label(reply).topic == "second"
    reply = '{"topic": "first"} then {"note": "no topic here"}'
    assert parse_label(reply).topic == "first"


def test_braces_inside_strings_do_not_split_groups():
    reply = '{"topic": "set {a}", "description": "uses } and { freely"}'
    assert parse_label(reply) == Label("set {a}", "uses } and { freely", [])


def test_duplicate_keywords_differing_in_case_collapse_to_the_first():
    reply = json.dumps({"topic": "bone", "keywords": ["Bone", "bone ", "BONE", "cells", "", "  "]})
    assert parse_label(reply).keywords == ["Bone", "cells"]


def test_keywords_are_capped_at_ten_and_non_strings_are_dumped():
    reply = json.dumps({"topic": "t", "keywords": [f"k{i}" for i in range(15)]})
    assert parse_label(reply).keywords == [f"k{i}" for i in range(10)]
    reply = json.dumps({"topic": "t", "keywords": [None, 1, {"k": "é"}, [2]]})
    assert parse_label(reply).keywords == ["null", "1", '{"k": "é"}', "[2]"]
    assert parse_label(json.dumps({"topic": "t", "keywords": "single"})).keywords == ["single"]
    assert parse_label(json.dumps({"topic": "t", "keywords": "   "})).keywords == []


def test_truncated_reply_is_recovered():
    reply = 'Reasoning...\n{"topic": "bone", "description": "This neuron reacts to bone tiss'
    assert parse_label(reply) == Label("bone", "This neuron reacts to bone tiss", [])
    reply = '{"topic": "bone", "description": "Bone.", "keywords": ["bone", "gra'
    assert parse_label(reply) == Label("bone", "Bone.", ["bone", "gra"])


def test_list_valued_topic_is_joined():
    assert parse_label('{"topic": ["bone", "cells"]}').topic == "bone; cells"


def test_null_fields_become_empty_strings():
    assert parse_label('{"topic": null, "description": null, "keywords": null}') == Label("", "", [])


@pytest.mark.parametrize(
    "obj",
    [
        {"description": "no topic"},
        {"topic": 5},
        {"topic": True},
        {"topic": {"a": 1}},
        {"topic": "t", "description": 3},
        {"topic": "t", "keywords": {"a": 1}},
        {"topic": "t", "keywords": 3},
        ["topic"],
    ],
)
def test_invalid_objects_are_rejected(obj):
    assert label_from_json(obj) is None
    reply = json.dumps(obj)
    label = parse_label(reply)
    assert label.topic == PARSE_ERROR
    assert label.description == reply


def test_reply_without_json_is_a_parse_error_keeping_the_reply():
    assert parse_label("I cannot tell.") == Label(PARSE_ERROR, "I cannot tell.", [])


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"score": 0.3}', (0.3, 0.3)),
        ('Sure.\n```json\n{"score": 0.25}\n```', (0.25, 0.25)),
        ('{"score": "0.7"}', (0.7, 0.7)),
        ('{"score": 1.4}', (1.0, 1.4)),
        ('{"score": -0.2}', (0.0, -0.2)),
        ('{"score": 0.2} then {"note": 1}', (None, None)),
        ("score: 0.4", (None, None)),
        ('{"score": null}', (None, None)),
        ('{"score": "high"}', (None, None)),
        ('{"probability": 0.4}', (None, None)),
        ('{"score": {"value": 0.4}}', (None, None)),
    ],
)
def test_score_parsing_cases(reply, expected):
    assert parse_score(reply) == expected


def test_score_parsing_keeps_the_raw_value_of_non_finite_numbers():
    score, raw = parse_score('{"score": NaN}')
    assert score == 1.0 and math.isnan(raw)
